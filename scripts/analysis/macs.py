from srpaper import runtime as rt


def run(args):
    import torch
    from srpaper.utils.macs import MACCounter
    rt.inference_precision()
    out=rt.new_run("macs")
    rows=[]
    for label in args.models:
        spec=rt.model_specs()[label]
        height,width={2:(360,640),3:(240,426),4:(180,320)}[spec["scale"]]
        model,record=rt.load_model(label,args.device,args.checkpoint_dir)
        for mode in args.inference:
            if spec["composition"]=="fair" and mode=="TLC":continue
            rt.set_tlc(model,mode=="TLC")
            counter=MACCounter()
            counter.attach(model)
            try:
                with torch.inference_mode():
                    model(torch.zeros(1,3,height,width,device=args.device))
                rows.append({"model":label,"scale":spec["scale"],"inference":mode,"input_height":height,"input_width":width,
                             "params":spec["params"],"macs":counter.total_macs,"gmacs":counter.total_macs/1e9,
                             "conv_macs":counter.conv_macs,"linear_macs":counter.linear_macs,
                             "dynamic_filter_macs":counter.dynamic_filter_macs,"attention_matmul_macs":counter.attention_matmul_macs})
            finally:
                counter.detach()
    rt.write_csv(out/"macs.csv",rows)
    print(rows)
    print("Results:",out)
