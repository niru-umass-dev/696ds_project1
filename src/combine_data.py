import json


with open("data/dev.json") as f:
    dev = json.load(f)
with open("data/test.json") as f:
    test = json.load(f)
with open("gen/cont/codec/outputs.json") as f:
    gen_cont = json.load(f)
with open("gen/comm/codec/outputs.json") as f:
    gen_comm = json.load(f)

out = dict(dev=list(), test=list())
for idx, pair in enumerate(dev):
    out_dic = pair
    out_dic['gen_cont_a'] = gen_cont['dev'][idx * 2]['prediction']
    out_dic['gen_cont_b'] = gen_cont['dev'][idx * 2 + 1]['prediction']
    out_dic['gen_comm_a'] = gen_comm['dev'][idx * 2]['prediction']
    out_dic['gen_comm_b'] = gen_comm['dev'][idx * 2 + 1]['prediction']
    out['dev'].append(out_dic)
for idx, pair in enumerate(test):
    out_dic = pair
    out_dic['gen_cont_a'] = gen_cont['test'][idx * 2]['prediction']
    out_dic['gen_cont_b'] = gen_cont['test'][idx * 2 + 1]['prediction']
    out_dic['gen_comm_a'] = gen_comm['test'][idx * 2]['prediction']
    out_dic['gen_comm_b'] = gen_comm['test'][idx * 2 + 1]['prediction']
    out['test'].append(out_dic)

with open("data/combined.json", "w") as f:
    json.dump(out, f, indent=4)
