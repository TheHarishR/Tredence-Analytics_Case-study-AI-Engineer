# Short Report: Self-Pruning Neural Network

## Why An L1 Penalty On Sigmoid Gates Encourages Sparsity

Each gate is produced by applying a sigmoid to a learnable score, so every gate value stays between `0` and `1`. Adding an L1 penalty on these gate values makes the optimizer pay a cost whenever many gates remain active. Because smaller gates reduce that penalty, the model is encouraged to push less useful gates toward `0`. Once a gate becomes very small, the corresponding connection or neuron contributes little to the final prediction, which makes the learned network sparse.

## Result Summary

The executed notebook shows the expected sparsity-vs-accuracy trade-off across multiple regularization strengths:

| Lambda | Test Accuracy | Sparsity Level (%) |
| --- | ---: | ---: |
| 0.0001 | 51.58 | 19.98 |
| 0.0003 | 50.99 | 31.25 |
| 0.0010 | 37.88 | 57.59 |
| 0.0030 | 13.84 | 84.82 |

These values come from the compact hard-pruned model produced by the executed notebook. They show the intended behavior clearly:

- lower `lambda` keeps more capacity and preserves accuracy
- higher `lambda` produces stronger pruning
- excessive pruning eventually hurts performance

## Best Model

Based on the notebook outputs, the best practical trade-off is:

- `Lambda = 0.0001`
- `Test Accuracy = 51.58%`
- `Sparsity Level = 19.98%`

## Gate Distribution Plot


<img width="1149" height="585" alt="image" src="https://github.com/user-attachments/assets/6cf3f826-1145-4734-92d2-a140491fb6d5" />


The histogram shows that a significant portion of gates are pushed close to zero while another group remains active, which is exactly the pattern expected from successful self-pruning.
