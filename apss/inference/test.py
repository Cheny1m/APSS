import mindspore
mindspore.set_context(device_target="GPU", device_id = 1)



from apss.nets.attention_model_v2 import AttentionStepModel
from apss.problems import pp
import mindspore.ops

import os
import json
a = mindspore.ops.rand([1000,7,3])
checkpoint_path = "/root/APSS/checkpoint/AttentionModelV2"


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args

def load_problem(name):
    from apss.problems import PP
    problem = {
        'pp': PP
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem
problem = load_problem("pp")

path = "/root/APSS/checkpoint/AttentionModelV2/pp_30_2"

args = load_args(os.path.join(path, 'args.json'))


model = AttentionStepModel(
    args['embedding_dim'],
    args['hidden_dim'],
    problem,
    n_encode_layers=args['n_encode_layers'],
    mask_inner=True,
    mask_logits=True,
    normalization=args['normalization'],
    tanh_clipping=args['tanh_clipping'],
    checkpoint_encoder=args.get('checkpoint_encoder', False),
    shrink_size=args.get('shrink_size', None),
    num_split=args.get('num_split', None),
    node_size=args.get('node_size', None)
)
dic = mindspore.load_checkpoint(os.path.join(checkpoint_path,"pp_30_2/pp_30_2_final.ckpt"))
mindspore.load_param_into_net(model,dic)


model.set_train(False)
print(model(a))