# ResNet

## Why deep?

1. Learn more complex representation of data
2. "Level" of features can be enriched by the number of stacked layers

## What problem that deep has?

1. Gradient vanishing - due to backpropagation - multiple multiplies, if weight is small, multi-multiplies will cause gradient smaller and smaller
2. Degradation problem - deeper net has higher **training** error

![](<.gitbook/assets/image (3).png>)



## Residual block

_The skip connections are essentially restoring the identity of the input from the first layer of the block, thus keeping the block output similar to the input_

Choose of number of layer skipped: case by case

![](<.gitbook/assets/image (34).png>)



## Plain network

![](<.gitbook/assets/image (13).png>)

Shortcut/skip connection

![](<.gitbook/assets/image (26).png>)



## Why residual network?

![](<.gitbook/assets/image (9).png>)

W and b = 0 because L2 regulization.&#x20;

![](<.gitbook/assets/image (23).png>)

![](<.gitbook/assets/image (30).png>)
