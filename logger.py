import logging
from datetime import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from glob import glob


class DFLogger:
    """
    column_names: can be a string then it is splitted on the separator, else it can be a list of strings
    log_dir_suffix: string to append as suffix to the log directory, default is empty string
    figure_ext: string indicating the extension of the plots to be saved, default is '.png'.
                Note that pdfs are always saved. This is extra if you need another format.


    """

    def __init__(
        self,
        column_names,
        logger_name="df_logger",
        sep="|",
        log_filename="log.txt",
        log_dir_suffix="",
        linewidth=3,
        plot_font_size=18,
        plot_style="tableau-colorblind10",
        figure_size=(10, 8),
        copy_config_files=True,
        files_to_copy_start_with=("1_", "2_"),
        plot_ungrouped=("lr", "epoch"),
        figure_ext=".png",
    ):
        self.column_names = []
        self.SEP = sep
        self.linewidth = linewidth
        self.plot_font_size = plot_font_size
        self.figure_size = figure_size
        self.copy_config_files = copy_config_files
        self.files_to_copy_start_with = files_to_copy_start_with
        self.plot_ungrouped = plot_ungrouped
        self.figure_ext = figure_ext

        # make sure style is available, else use 1st available one
        available_styles = plt.style.available
        self.plot_style = (
            available_styles[0] if plot_style not in available_styles else plot_style
        )

        self.parse_column_names(column_names)

        self.log_filename = log_filename
        self.log_dir_suffix = log_dir_suffix
        self.log_dir = os.path.join(
            "./logs",
            datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S-{self.log_dir_suffix}"),
        )

        # create plots folder for this run
        self.plots_directory = self.log_dir
        if not os.path.exists(self.plots_directory):
            os.makedirs(self.plots_directory)

        self.log_filepath = os.path.join(self.log_dir, self.log_filename)

        self.current_index = 0
        self.n = len(self.column_names)

        self.current_line = ""

        self.df_logger = logging.getLogger(logger_name)
        self.df_logger.setLevel(logging.INFO)
        self.df_logger.addHandler(logging.FileHandler(self.log_filepath, "w+"))

        if copy_config_files:
            self.copy_config_files_to_log_dir()

        # add header columns
        self.df_logger.info(self.SEP.join(self.column_names))

        # set plotting style, figure size, font size
        plt.style.use(self.plot_style)
        plt.rcParams.update(
            {"font.size": self.plot_font_size, "figure.figsize": self.figure_size}
        )

    def copy_config_files_to_log_dir(self):
        files_list = glob("*.py")
        for file in files_list:
            if file.startswith(tuple(self.files_to_copy_start_with)):
                print("Script file: ", file, " copied to log directory.")
                print("-" * 50)
                copyfile(file, os.path.join(self.plots_directory, file))

    def save_val_predictions(self, df, f1=False):
        filesuffix = "_f1.csv" if f1 else "_loss.csv"
        df.to_csv(
            os.path.join(self.plots_directory, "best_val_predictions" + filesuffix),
            index=False,
        )

        filesuffix = "_f1.pkl" if f1 else "_loss.pkl"
        df.to_pickle(
            os.path.join(self.plots_directory, "best_val_predictions" + filesuffix)
        )

    def parse_column_names(self, column_names):
        # column names provided as a string with the defined separator used
        if isinstance(column_names, str):
            self.column_names = column_names.split(self.SEP)

        # column names provided as a list of strings
        if isinstance(column_names, list):
            self.column_names = column_names

        # remove any spaces in column names and make sure they are strings
        self.column_names = [c.strip() for c in self.column_names]

    def log_value(self, value):
        self.current_line += str(value)
        self.current_index += 1
        if self.current_index == self.n:
            self.df_logger.info(self.current_line)
            self.current_index = 0
            self.current_line = ""
        else:
            self.current_line += self.SEP

    def log_values_list(self, values, screen_print=False):
        assert (
            len(values) == self.n
        ), "Logging a list of values which is not the same size as the number of columns of the logger"
        assert (
            self.current_index == 0
        ), "Logging a line while the pointer is not at the start of a line"
        values = [str(x) for x in values]
        for v in values:
            self.log_value(v)
        if screen_print:
            print(" " + self.SEP + " ".join(values))
        return True

    def read_current_log_file(self) -> pd.DataFrame:
        return pd.read_csv(self.log_filepath, sep=self.SEP)

    def plot_columns(
        self, flag_column="flag", batches_per_epoch=0, epoch=0, log_interval=0
    ):
        df = self.read_current_log_file()
        group = True

        if flag_column not in df.columns:
            print(
                "Flag column name {} is not one of the columns of the logged df file."
                "Plotting without grouping.",
                flag_column,
            )
            group = False

        for c in df.columns:
            # exclude non-numeric columns of dataframe
            if not np.issubdtype(df[c].dtype, np.number):
                print("Skipping plotting non-numeric column {}.", c)
                continue

            # remove '/' and '\' from columns names since they disturb the path
            clean_c = c.replace("/", "_").replace("\\", "_")

            plot_filepath = os.path.join(
                self.plots_directory, clean_c + self.figure_ext
            )

            try:
                if group and c not in self.plot_ungrouped:
                    df.groupby(flag_column)[c].plot(
                        title=c, legend=True, linewidth=self.linewidth
                    )
                else:
                    if flag_column in df.columns:
                        df[df[flag_column] == "train"][c].plot(
                            title=c, legend=True, linewidth=self.linewidth
                        )
                    else:
                        df[c].plot(title=c, legend=True, linewidth=self.linewidth)

                if batches_per_epoch and epoch and log_interval:
                    xlimits = plt.axes().get_xlim()
                    for line_position in range(
                        0, int(xlimits[1]), batches_per_epoch // log_interval
                    ):
                        plt.axvline(x=line_position, color="gray", linestyle="--")
                plt.savefig(plot_filepath)
                plt.savefig(os.path.splitext(plot_filepath)[0] + ".pdf")
                plt.close()
            except ValueError:
                print("Column {} produced an error during plotting.", c)

            val_df = df[df[flag_column] == "val"]
            print(
                "Column (", c, ") statistics: ", min(val_df[c]), " -- ", max(val_df[c])
            )
