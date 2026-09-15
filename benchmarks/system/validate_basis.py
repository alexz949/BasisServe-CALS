"""Audit all fitted layers: invertibility, Base prefix and decoded output."""
import argparse
import os
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from basisserve.core.routing_basis import make_routing_basis
from benchmarks.system.common import metadata, save


def rel(actual, expected):
    return float((actual.double()-expected.double()).square().sum()/expected.double().square().sum().clamp_min(1e-30))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    a=p.parse_args();rank=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    torch.cuda.set_device(rank);torch.manual_seed(9000+rank);torch.set_num_threads(2)
    rows=[]
    for layer in range(rank,32,world):
        factors=load_file(str(a.root/'v96/selected_factors'/f'layer_{layer:03d}.safetensors'),device=f'cuda:{rank}')
        router=load_file(str(a.root/'ours_b16r16'/f'layer_{layer:03d}.safetensors'),device=f'cuda:{rank}')
        ranks=factors['source_ranks'];assert torch.all(ranks==ranks[0])
        width=int(ranks[0]);enc=factors['value_coordinate_encoders'][:,:,:width].double()
        dec=factors['head_output_decoders'][:,:width].double();left=router['base_left_b16'].double()
        assert left.shape==(8,width,16)
        basis=make_routing_basis(left);v=torch.randn(8,64,128,device='cuda',dtype=torch.float64)
        original=v@enc;transformed=v@basis.encoder(enc)
        base,payload=basis.split(original)
        torch.testing.assert_close(base,original@left,rtol=1e-9,atol=1e-9)
        torch.testing.assert_close(transformed,torch.cat((base,payload),-1),rtol=1e-9,atol=1e-9)
        original_output=original.repeat_interleave(4,0)@dec
        transformed_output=transformed.repeat_interleave(4,0)@basis.decoder(dec)
        torch.testing.assert_close(transformed_output,original_output,rtol=1e-8,atol=1e-8)
        assert base.is_contiguous() and payload.is_contiguous()
        assert base.untyped_storage().data_ptr()!=payload.untyped_storage().data_ptr()
        old_bf=v.bfloat16()@enc.bfloat16();new_bf=v.bfloat16()@basis.encoder(enc).bfloat16()
        old_out=old_bf.repeat_interleave(4,0)@dec.bfloat16()
        new_out=new_bf.repeat_interleave(4,0)@basis.decoder(dec).bfloat16()
        row=dict(layer=layer,v_rank=width,base_rank=16,residual_rank=16,
                 max_condition=float(basis.condition.max()),fp64_output_rel_mse=rel(transformed_output,original_output),
                 bf16_output_rel_mse=rel(new_out,old_out),bf16_base_rel_mse=rel(new_bf[:,:,:16],old_bf@left.bfloat16()),
                 separate_storage=True,logical_router_dimensions=32)
        rows.append(row);print(row,flush=True)
        target=a.output/'routing_basis'/f'layer_{layer:03d}.safetensors';target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists()
        save_file(dict(transform=basis.transform.cpu(),inverse=basis.inverse.cpu().contiguous(),
                       value_coordinate_encoders=basis.encoder(enc).float().cpu(),
                       head_output_decoders=basis.decoder(dec).float().cpu()),str(target))
    save(a.output/f'basis_validation_rank{rank}.json',dict(metadata=metadata(),rows=rows,
        scope='Synthetic V inputs, real fitted factors, all assigned layers; not real-capture or model quality validation'))

if __name__=='__main__':main()
