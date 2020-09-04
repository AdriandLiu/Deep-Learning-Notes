# Transfer Learning

{% embed url="https://www.youtube.com/watch?v=yofjFQddwHE" %}

0:00 - 1:25

Use a pre-trained model and apply it in different task by fine-tuning \(add a classification layer or etc.\)

For example, we trained a DL model to classify cat and dog, we can exclude the last pre-trained layer and add a new layer for a new task, say, X-ray scans. We have to use X-ray scans do fine-tuning the pre-trained model to "transfer" the knowledge to the new task.

Reason:

X-ray scans may not have sufficient data for DL model to train, but cat and dog data are available for large training. After obtaining the pre-trained DL model, get rid of the last layer and fine-tune the model with new task's data. 

![](.gitbook/assets/image%20%2855%29.png)



