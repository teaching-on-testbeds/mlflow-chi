
::: {.cell .markdown}

# ML Experiment Tracking with MLFlow

In this tutorial, we explore some of the infrastructure and platform requirements for large model training, and to support the training of many models by many teams. We focus specifically on experiment tracking (using [MLFlow](https://mlflow.org/)).

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@TACC site.

:::

::: {.cell .markdown}

## Experiment resources 

For this experiment, we will provision one bare-metal node with GPUs. 

The MLFlow experiment is more interesting if we run it on a node with two GPUs, because then we can better understand how to configure logging in a distributed training run. But, if need be, we can run it on a node with one GPU.

We can browse Chameleon hardware configurations for suitable node types using the [Hardware Browser](https://chameleoncloud.org/hardware/). For example, to find nodes with 2x GPUs: if we expand "Advanced Filters", check the "2" box under "GPU count", and then click "View", we can identify some suitable node types. 

:::

::: {.cell .markdown .gpu-nvidia}

In this version of the lab, we use NVIDIA `compute_liqid` nodes at CHI@TACC.

* The `compute_liqid` nodes at CHI@TACC have one or two NVIDIA A100 40GB GPUs. As of this writing, `liqid01` and `liqid02` have two GPUs.
* Alternatively, to use `gpu_mi100` nodes and follow the AMD version of this lab, refer to the [AMD instructions](index_amd).

:::

::: {.cell .markdown .gpu-amd}

In this version of the lab, we use AMD `gpu_mi100` nodes at CHI@TACC.

* Most of the `gpu_mi100` nodes have two AMD MI100 GPUs.
* Alternatively, to use `compute_liqid` nodes and follow the NVIDIA version of this lab, refer to the [NVIDIA instructions](index_nvidia).

:::

::: {.cell .markdown .gpu-nvidia}

To use NVIDIA instructions directly, refer to the [NVIDIA instructions](index_nvidia).

:::

::: {.cell .markdown .gpu-amd}

To use AMD instructions directly, refer to the [AMD instructions](index_amd).

:::

::: {.cell .markdown}

Once you decide on GPU type, make sure to follow the instructions specific to that GPU type.


:::

::: {.cell .markdown}

Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your time.

:::

::: {.cell .markdown}

At the beginning of your lease time, continue with `1_create_lease.ipynb`, then continue with the create-server notebook for your GPU type.

:::

::: {.cell .markdown .gpu-nvidia}

For NVIDIA runs, refer to the [NVIDIA instructions](index_nvidia), then continue with `1_create_server.ipynb`.

:::

::: {.cell .markdown .gpu-amd}

For AMD runs, refer to the [AMD instructions](index_amd), then continue with `1_create_server.ipynb`.

:::
