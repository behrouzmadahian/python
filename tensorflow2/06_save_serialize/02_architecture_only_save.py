"""
Sometime you are interested in architecutre of the model and you don't need to save the weights or optimizer.
- In this case you can retrieve the "config" of the model via get_config() method
- The config is a Python dict that enables you to re-create the same model -- initialized from scratch,
without any of the information learned previously during training
"""
