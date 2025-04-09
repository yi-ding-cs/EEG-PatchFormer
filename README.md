# EEG-PatchFormer
This is the PyTorch implementation of EEG-PatchFormer in our paper:

Ding, Yi, Joon Hei Lee, Shuailei Zhang, Tianze Luo, and Cuntai Guan. "Decoding Human Attentive States from Spatial-temporal EEG Patches Using Transformers." in the 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). PDF availible [here](https://arxiv.org/abs/2502.03736)
## Example of the usage
```python
from EEG_PatchFormer import PatchFormer

original_order = ['Fp1', 'AFF5', 'AFz', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 'P7', 'P3',
                      'Pz', 'POz', 'O1', 'Fp2', 'AFF6', 'F2', 'FC2', 'FC6', 'C4', 'T8', 'CP2', 'CP6', 'P4',
                      'P8', 'O2']

graph_general = [['Fp1', 'Fp2'], ['AFF5', 'AFz', 'AFF6'], ['F1', 'F2'],
                ['FC5', 'FC1', 'FC2', 'FC6'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                ['P7', 'P3', 'Pz', 'P4', 'P8'], ['POz'], ['O1', 'O2'],
                ['T7'], ['T8']]

graph_idx = graph_general  # The general graph definition.
idx = []
num_chan_local_graph = []
for i in range(len(graph_idx)):
    num_chan_local_graph.append(len(graph_idx[i]))
    for chan in graph_idx[i]:
        idx.append(original_order.index(chan))

data = torch.randn(1, 28, 800)  # (batch_size=1, EEG_channel=28, data_points=800)
data = data[:, idx, :]  # (batch_size=1, EEG_channel=28, data_points=800)

net = PatchFormer(
    num_classes=2, input_size=(1, 28, 800), sampling_rate=200, num_T=32, patch_time=20, patch_step=5,
    dim_head=32, depth=4, heads=32,
    dropout_rate=0.5, idx_graph=num_chan_local_graph)
print(net)
print(count_parameters(net))

out = net(data)
```

# CBCR License
| Permissions | Limitations | Conditions |
| :---         |     :---:      |          :---: |
| :white_check_mark: Modification   | :x: Commercial use   | :warning: License and copyright notice   |
| :white_check_mark: Distribution     |       |      |
| :white_check_mark: Private use     |        |      |

# Cite
Please cite our paper if you use our code in your own work:

```
@misc{ding2025decodinghumanattentivestates,
      title={Decoding Human Attentive States from Spatial-temporal EEG Patches Using Transformers}, 
      author={Yi Ding and Joon Hei Lee and Shuailei Zhang and Tianze Luo and Cuntai Guan},
      year={2025},
      eprint={2502.03736},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2502.03736}, 
}
```
