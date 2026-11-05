"""Exact CPU tensor identity for portable repack of the frozen INT8 codec."""
import hashlib

def tensor_manifest(packed):
 result={}
 for name,fields in packed['model'].items():
  result[name]={}
  for key,value in fields.items():
   assert value.device.type=='cpu'
   v=value.detach().contiguous()
   result[name][key]=dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest())
 return result

def validate(packed,receipt):
 assert packed['storage_format']==receipt['storage_format']
 assert packed['quantized_names']==receipt['quantized_names']
 assert tensor_manifest(packed)==receipt['tensor_manifest'],'REPACK_NOT_EXACT_ORIGINAL_INT8'
 for key in ('protocol_sha256','pretrained_sha256','train_users','held_users','epochs','seed'):
  assert packed[key]==receipt['checkpoint_metadata'][key],key
