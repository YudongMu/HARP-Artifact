"""DiT instance adapter: no edits to the upstream models.py file."""
import types
import torch
import torch.nn.functional as F

class DiTAdapter:
 def __init__(self,model,runtime):self.model=model;self.runtime=runtime;self.handles=[];self.original=[]
 def attach(self):
  if self.original:raise RuntimeError('Adapter already attached')
  for layer,block in enumerate(self.model.blocks):
   attn=block.attn;original=attn.forward;self.original.append((attn,original))
   runtime=self.runtime
   def forward(this,x,layer=layer):
    b,n,c=x.shape
    q,k,v=this.qkv(x).reshape(b,n,3,this.num_heads,c//this.num_heads).permute(2,0,3,1,4).unbind(0)
    q,k=this.q_norm(q),this.k_norm(k)
    if runtime.observer is not None:
     for site,tensor in zip(['q','k','v'],[q,k,v]):
      bq,hq,nq,dq=tensor.shape;runtime.observer.record(runtime,tensor.reshape(bq*hq,nq,dq),layer,site)
     runtime.observer.attention(runtime,q,k,v,layer)
    if runtime.mode in {'residual','combined'}:
     transform=(lambda site,t:runtime.process_qkv(site,t,layer)) if runtime.mode=='combined' or runtime.intervention is not None else None
     y=runtime.attention(q,k,v,runtime.step,('dit',layer),token_order=runtime.order,active_transform=transform)
     runtime.stats['attention_calls']+=1
     runtime.stats['last_attention']=runtime.attention.last_stats
     runtime.stats['attention_counters']=dict(runtime.attention.counters)
     totals=runtime.stats.setdefault('attention_totals',{})
     for key in ['selected_region_pairs','candidate_region_pairs','residual_elements_read','residual_elements_written']:
      totals[key]=totals.get(key,0)+runtime.attention.last_stats[key]
     totals['selected_region_ratio_weighted']=totals['selected_region_pairs']/totals['candidate_region_pairs']
    else:
     if runtime.mode=='planeclear':q,k,v=[runtime.process_qkv(site,t,layer) for site,t in zip(['q','k','v'],[q,k,v])]
     y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
    return this.proj_drop(this.proj(y.transpose(1,2).reshape(b,n,c)))
   attn.forward=types.MethodType(forward,attn)
   for module,site in [(attn.qkv,'qkv_input'),(attn.proj,'o_proj_input'),(block.mlp.fc1,'ffn1_input'),(block.mlp.fc2,'ffn2_input')]:
    def before(module,args,layer=layer,site=site):return (runtime.process_tokens(args[0],layer,site),*args[1:])
    self.handles.append(module.register_forward_pre_hook(before))
  return self
 def close(self):
  for handle in self.handles:handle.remove()
  for module,original in self.original:module.forward=original
  self.handles.clear();self.original.clear();self.runtime.reset()
 def __enter__(self):return self.attach()
 def __exit__(self,*args):self.close()
