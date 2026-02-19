// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use anyhow::Result;
use nixl_sys::{MemoryRegion, NixlDescriptor, XferDescList, XferStatus};
use std::future::Future;

type Addr = usize;
type DeviceId = u64;
type Size = usize;

/// Maximum size for a single transfer entry in bytes (vast_dynamo aggregation)
const MAX_TRANSFER_SIZE: usize = 4 * 1024 * 1024; // 4MB

#[derive(Debug)]
struct XferAggr {
    descriptors: Vec<(Addr, Addr, Size, DeviceId, DeviceId)>, // (src_addr, dst_addr, size, src_device_id, dst_device_id)
}

impl XferAggr {
    fn new() -> Self {
        Self { descriptors: Vec::new() }
    }

    fn add_desc(
        &mut self,
        src: Addr,
        dst: Addr,
        size: Size,
        src_device_id: DeviceId,
        dst_device_id: DeviceId,
    ) {
        self.descriptors
            .push((src, dst, size, src_device_id, dst_device_id));
    }

    fn populate_xfer_desc_list(
        &mut self,
        src_dl: &mut XferDescList,
        dst_dl: &mut XferDescList,
    ) -> Result<()> {
        // Sort by (src_device_id, dst_device_id, src_addr) for better grouping
        self.descriptors.sort_by_key(
            |&(src_addr, _dst_addr, _size, src_device_id, dst_device_id)| {
                (src_device_id, dst_device_id, src_addr)
            },
        );

        let mut pending: Option<(Addr, Addr, Size, DeviceId, DeviceId)> = None;

        for (src, dst, size, src_device_id, dst_device_id) in self.descriptors.drain(..) {
            if let Some((last_src, last_dst, last_size, last_src_device_id, last_dst_device_id)) =
                &mut pending
            {
                // Merge contiguous descriptors when possible
                if *last_src_device_id == src_device_id
                    && *last_dst_device_id == dst_device_id
                    && *last_src + *last_size == src
                    && *last_dst + *last_size == dst
                    && *last_size + size <= MAX_TRANSFER_SIZE
                {
                    *last_size += size;
                    continue;
                } else {
                    // Flush pending
                    Self::add_chunked_descriptors(
                        src_dl,
                        dst_dl,
                        *last_src,
                        *last_dst,
                        *last_size,
                        *last_src_device_id,
                        *last_dst_device_id,
                    )?;
                }
            }

            pending = Some((src, dst, size, src_device_id, dst_device_id));
        }

        if let Some((last_src, last_dst, last_size, last_src_device_id, last_dst_device_id)) =
            pending
        {
            Self::add_chunked_descriptors(
                src_dl,
                dst_dl,
                last_src,
                last_dst,
                last_size,
                last_src_device_id,
                last_dst_device_id,
            )?;
        }

        Ok(())
    }

    /// Add descriptors to the transfer lists, chunking large transfers if they exceed MAX_TRANSFER_SIZE
    fn add_chunked_descriptors(
        src_dl: &mut XferDescList,
        dst_dl: &mut XferDescList,
        mut src_addr: Addr,
        mut dst_addr: Addr,
        mut size: Size,
        src_device_id: DeviceId,
        dst_device_id: DeviceId,
    ) -> Result<()> {
        while size > 0 {
            let chunk_size = size.min(MAX_TRANSFER_SIZE);
            src_dl.add_desc(src_addr, chunk_size, src_device_id);
            dst_dl.add_desc(dst_addr, chunk_size, dst_device_id);

            src_addr += chunk_size;
            dst_addr += chunk_size;
            size -= chunk_size;
        }
        Ok(())
    }
}

fn append_xfer_request<Source, Destination>(
    src: &Source,
    dst: &mut Destination,
    xfer_aggr: &mut XferAggr,
) -> Result<()>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    let src_data = src.block_data();
    let dst_data = dst.block_data_mut();

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_desc = src_data.block_view()?.as_nixl_descriptor();
        let dst_desc = dst_data.block_view_mut()?.as_nixl_descriptor_mut();

        // Extract device IDs from both descriptors
        let src_device_id = src_desc.device_id();
        let dst_device_id = dst_desc.device_id();

        unsafe {
            xfer_aggr.add_desc(
                src_desc.as_ptr() as usize,
                dst_desc.as_ptr() as usize,
                src_desc.size(),
                src_device_id,
                dst_device_id,
            );
        }

        Ok(())
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        for layer_idx in 0..src_data.num_layers() {
            for outer_idx in 0..src_data.num_outer_dims() {
                let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

                debug_assert_eq!(src_view.size(), dst_view.size());

                let src_desc = src_view.as_nixl_descriptor();
                let dst_desc = dst_view.as_nixl_descriptor_mut();

                let src_device_id = src_desc.device_id();
                let dst_device_id = dst_desc.device_id();

                unsafe {
                    xfer_aggr.add_desc(
                        src_desc.as_ptr() as usize,
                        dst_desc.as_ptr() as usize,
                        src_desc.size(),
                        src_device_id,
                        dst_device_id,
                    );
                }
            }
        }
        Ok(())
    }
}

/// Copy a block from a source to a destination using CUDA memcpy
pub fn write_blocks_to<Source, Destination>(
    src: &[Source],
    dst: &mut [Destination],
    ctx: &Arc<TransferContext>,
    transfer_type: NixlTransfer,
) -> Result<Box<dyn Future<Output = ()> + Send + Sync + Unpin>>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    if src.is_empty() || dst.is_empty() {
        return Ok(Box::new(std::future::ready(())));
    }
    assert_eq!(src.len(), dst.len());

    let nixl_agent_arc = ctx.as_ref().nixl_agent();
    let nixl_agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .expect("NIXL agent not found");

    let src_mem_type = src
        .first()
        .unwrap()
        .block_data()
        .storage_type()
        .nixl_mem_type();
    let dst_mem_type = dst
        .first()
        .unwrap()
        .block_data()
        .storage_type()
        .nixl_mem_type();

    let mut src_dl = XferDescList::new(src_mem_type)?;
    let mut dst_dl = XferDescList::new(dst_mem_type)?;

    // Aggregate contiguous descriptors before adding to XferDescList
    let mut xfer_aggr = XferAggr::new();

    for (src, dst) in src.iter().zip(dst.iter_mut()) {
        append_xfer_request(src, dst, &mut xfer_aggr)?;
    }

    xfer_aggr.populate_xfer_desc_list(&mut src_dl, &mut dst_dl)?;

    let xfer_req = nixl_agent.create_xfer_req(
        transfer_type.as_xfer_op(),
        &src_dl,
        &dst_dl,
        &nixl_agent.name(),
        None,
    )?;

    let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;

    if still_pending {
        Ok(Box::new(Box::pin(async move {
            let nixl_agent = nixl_agent_arc
                .as_ref()
                .as_ref()
                .expect("NIXL agent not found");

            loop {
                match nixl_agent.get_xfer_status(&xfer_req) {
                    Ok(XferStatus::Success) => break, // Transfer is complete.
                    Ok(XferStatus::InProgress) => {
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await
                    } // Transfer is still in progress.
                    Err(e) => {
                        tracing::error!("Error getting transfer status: {}", e);
                        break;
                    }
                }
            }
        })))
    } else {
        Ok(Box::new(std::future::ready(())))
    }
}
