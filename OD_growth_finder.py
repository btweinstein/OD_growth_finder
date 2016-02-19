import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import datetime as datetime

class OD_growth_experiment(object):

    def __init__(self, path_to_data, output_path = './', s=0.2, constant_background=0):
        self.path_to_data = path_to_data
        self.data = pd.read_excel(path_to_data)
        # Drop the rows that have NAN's, usually at the end
        self.data.dropna(inplace=True, axis=1)

        # Get the times from the data
        times = self.data.loc[:, 'Time']
        self.elapsed_minutes = times.values

        # Set the output path
        self.output_path = output_path

        # Set the default s for fitting...deals with how close the fit is to the points
        self.s = s
        self.constant_background = constant_background

    def get_max_growth_rate(self, well_str):
        data_to_use = np.log(self.data.loc[:, well_str] - self.constant_background).values # Log of the OD
        # Drop the negative infinities...indistinguishable from noise
        to_keep = np.isfinite(data_to_use)

        data_to_use = data_to_use[to_keep]
        elapsed_minutes = self.elapsed_minutes[to_keep]

        interpolator = sp.interpolate.UnivariateSpline(elapsed_minutes, data_to_use, k=5, s=self.s)
        der = interpolator.derivative()

        # Get the approximation of the derivative at all points
        der_approx = der(elapsed_minutes)

        # Get the maximum
        maximum_index = np.argmax(der_approx)
        maximum_log_slope = der_approx[maximum_index]
        maximum_time = elapsed_minutes[maximum_index]

        return maximum_log_slope, maximum_time, maximum_index + np.sum(~to_keep)

    def plot_raw_data(self, well_str):
        data_to_use = self.data.loc[:, well_str] - self.constant_background

        plt.plot(self.elapsed_minutes, data_to_use, ls='', marker='.', label='Raw Data')

        plt.xlabel('Elapsed Time (minutes)')
        plt.ylabel(r'OD600')

        plt.legend(loc='best')

    def plot_growth_prediction(self, well_str, minutes_around_max=100):
        maximum_log_slope, maximum_time, maximum_index = self.get_max_growth_rate(well_str)

        data_to_use = np.log(self.data.loc[:, well_str] - self.constant_background) # Log of the OD

        plt.plot(self.elapsed_minutes, data_to_use, ls='', marker='.', label='Raw Data')

        times_around_max = np.linspace(maximum_time - minutes_around_max, maximum_time + minutes_around_max)
        predicted = times_around_max * maximum_log_slope - maximum_log_slope*maximum_time
        # Add so that predicted is where you expect
        predicted += data_to_use.values[maximum_index]


        plt.plot(times_around_max, predicted, ls='-', label='Max Growth', color='red', alpha=0.5)

        plt.xlabel('Elapsed Time (minutes)')
        plt.ylabel(r'$\log$(OD600)')

        plt.legend(loc='best')

    def get_all_growth_rates(self, save_pictures=False):

        growth_rate_data = []

        for cur_col in self.data.columns:
            # Check if this is a column we want
            if cur_col[0].isalpha() and cur_col[1].isnumeric():
                cur_growth_rate_data = []
                cur_growth_rate_data.append(cur_col)
                maximum_log_slope, maximum_time, maximum_index = self.get_max_growth_rate(cur_col)
                cur_growth_rate_data.append(maximum_log_slope)
                cur_growth_rate_data.append(maximum_time)
                cur_growth_rate_data.append(maximum_index)

                growth_rate_data.append(cur_growth_rate_data)

                if save_pictures:
                    self.plot_growth_prediction(cur_col)
                    plt.savefig(cur_col + '.png', dpi=200, bbox_inches='tight')
                    plt.clf()

        df = pd.DataFrame(growth_rate_data, columns=['well', 'growth_rate', 'max_time', 'max_index'])
        df['doubling_time'] = np.log(2)/df['growth_rate']

        return df