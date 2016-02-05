import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import datetime as datetime

def get_elapsed_hours(x):
    """Assumes x is a datetime.time object."""
    time_in_seconds = (60.*60.*x.hour + 60*x.minute + x.second)
    return time_in_seconds/(60.*60.)


class OD_growth_experiment(object, s=0.05):

    def __init__(self, path_to_data, output_path = './'):
        self.path_to_data = path_to_data
        self.data = pd.read_excel(path_to_data)
        # Drop the rows that have NAN's, usually at the end
        self.data.dropna(inplace=True, axis=1)

        # Get the times from the data
        times = self.data.loc[:, 'Time']
        times = times.astype(datetime.time)
        self.elapsed_hours = times.apply(get_elapsed_hours)

        # Set the output path
        self.output_path = output_path

        # Set the default s for fitting...deals with how close the fit is to the points
        self.s = s

    def get_max_growth_rate(self, well_str):
        data_to_use = np.log(self.data.loc[:, well_str]) # Log of the OD
        interpolator = sp.interpolate.UnivariateSpline(self.elapsed_hours, data_to_use, k=5, s=self.s)
        der = interpolator.derivative()

        # Get the approximation of the derivative at all points
        der_approx = der(self.elapsed_hours)

        # Get the maximum
        maximum_index = np.argmax(der_approx)
        maximum_log_slope = der_approx[maximum_index]
        maximum_time = self.elapsed_hours.values[maximum_index]

        return maximum_log_slope, maximum_time, maximum_index

    def plot_growth_prediction(self, well_str, hours_around_max = 2):
        maximum_log_slope, maximum_time, maximum_index = self.get_max_growth_rate(well_str)

        data_to_use = np.log(self.data.loc[:, well_str]) # Log of the OD

        plt.plot(self.elapsed_hours, data_to_use, ls='', marker='.', label='Raw Data')

        times_around_max = np.linspace(maximum_time - hours_around_max, maximum_time + hours_around_max)
        predicted = times_around_max * maximum_log_slope - maximum_log_slope*maximum_time
        # Add so that predicted is where you expect
        predicted += data_to_use.values[maximum_index]


        plt.plot(times_around_max, predicted, ls='-', label='Max Growth', color='red', alpha=0.5)

        plt.xlabel('Elapsed Time (hours)')
        plt.ylabel(r'$\log$(OD)')

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

        return pd.DataFrame(growth_rate_data, columns=['well', 'growth_rate', 'max_time', 'max_index'])