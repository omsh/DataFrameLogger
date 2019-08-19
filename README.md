# DataFrameLogger

A basic Logger based on [Pandas](https://pandas.pydata.org) useful specifically for training neural networks or other iterative processes that log data sequentially.

- Import the logger class from the `logger.py` file as follows:

  `from logger import DFLogger`

- Initialize the logger in your code providing the names of the dataframe columns (e.g. in your component `__init__` method):

  `df_logger = DFLogger(column_names='epoch | batch | lr | loss | accuracy | flag')`

- In your training/evaluation code, you can pass the logged values one by one or as a list (same size as the number of columns provided at inistantiation of the logger):

  `df_logger.log_values_list([epoch_number, batch_number, current_lr, loss_value, accuracy_value, 'train'])`


- A call to the plotting function will generate plots for all the numerical columns with some options as grouping per flag. This call can be added for example when training neural networks at the end of each epoch.

  `df_logger.plot_columns(epoch=epoch_number, batches_per_epoch=train_data_size // batch_size, log_interval=logging_interval)`
