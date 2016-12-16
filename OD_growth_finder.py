import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import datetime as datetime

class OD_growth_experiment(object):

    def __init__(self, path_to_data, output_path = './', s=0.2, cutoff_OD =.05):
        self.path_to_data = path_to_data
        self.data = pd.read_excel(path_to_data)
        # Drop the rows that have NAN's, usually at the end
        self.data.dropna(inplace=True, axis=1)

        # Get the times from the data
        times = self.data.loc[:, 'Time']
        self.elapsed_minutes = times.values

        # Drop the times column for simplicity
        self.data.drop('Time', axis=1, inplace=True)

        # Set the output path
        self.output_path = output_path

        # Set the default s for fitting...deals with how close the fit is to the points
        self.s = s

        # You ignore values with an OD lower than this.
        self.cutoff_OD = cutoff_OD

    def get_max_growth_rate(self, well_str):
        data_to_use = self.data.loc[:, well_str].values # Log of the OD
        # Drop the zeros

        to_keep = data_to_use > self.cutoff_OD
        data_to_use = data_to_use[to_keep]
        elapsed_minutes = self.elapsed_minutes[to_keep]

        interpolator = sp.interpolate.UnivariateSpline(elapsed_minutes, data_to_use, k=5, s=self.s)
        der = interpolator.derivative(n=1)
        der_2 = interpolator.derivative(n=2)

        # Get the approximation of the derivative at all points
        der_approx = der(elapsed_minutes)
        der_approx_2 = der_2(elapsed_minutes)

        # Calculate the slope
        alpha = (1./data_to_use)*der_approx

        slope_of_alpha = der_approx_2/data_to_use - der_approx**2/data_to_use**2

        plt.plot(elapsed_minutes, alpha)
        plt.figure()
        plt.plot(elapsed_minutes, slope_of_alpha)

        print np.max(slope_of_alpha)
        if np.all(slope_of_alpha < 0):
            print 'Missed exponential phase... :('
            return np.nan, np.nan

        plt.figure()
        plt.plot(elapsed_minutes, der_approx)
        plt.figure()
        plt.plot(elapsed_minutes, der_approx_2)

        # Get the maximum
        maximum_index = np.argmax(alpha)
        maximum_log_slope = alpha[maximum_index]
        maximum_time = elapsed_minutes[maximum_index]

        return maximum_log_slope, maximum_time

    def plot_raw_data(self, well_str):
        data_to_use = self.data.loc[:, well_str]

        plt.plot(self.elapsed_minutes, data_to_use, ls='', marker='.', label='Raw Data')

        plt.xlabel('Elapsed Time (minutes)')
        plt.ylabel(r'OD600')

        plt.legend(loc='best')

    def plot_growth_prediction(self, well_str, minutes_around_max=100):
        maximum_log_slope, maximum_time= self.get_max_growth_rate(well_str)

        data_to_use = np.log(self.data.loc[:, well_str]) # Log of the OD...make sure background is subtracted
        # Drop the negative infinities...indistinguishable from noise

        plt.plot(self.elapsed_minutes, data_to_use, ls='', marker='.', label='Raw Data')

        times_around_max = np.linspace(maximum_time - minutes_around_max, maximum_time + minutes_around_max)
        predicted = times_around_max * maximum_log_slope - maximum_log_slope*maximum_time
        # Add so that predicted is where you expect
        maximum_index,  = np.where(self.elapsed_minutes == maximum_time)[0]
        predicted += data_to_use.values[maximum_index]


        plt.plot(times_around_max, predicted, ls='-', label='Max Growth', color='red', alpha=0.5)

        plt.xlabel('Elapsed Time (minutes)')
        plt.ylabel(r'$\log$(OD600)')

        plt.legend(loc='best')

        plt.ylim(-10, 1)

    def get_all_growth_rates(self, save_pictures=False):

        growth_rate_data = []

        for cur_col in self.data.columns:
            # Check if this is a column we want
            if cur_col[0].isalpha() and cur_col[1].isnumeric():
                cur_growth_rate_data = []
                cur_growth_rate_data.append(cur_col)
                maximum_log_slope, maximum_time = self.get_max_growth_rate(cur_col)
                cur_growth_rate_data.append(maximum_log_slope)
                cur_growth_rate_data.append(maximum_time)
                maximum_index,  = np.where(self.elapsed_minutes == maximum_time)[0]
                cur_growth_rate_data.append(maximum_index)

                growth_rate_data.append(cur_growth_rate_data)

                if save_pictures:
                    self.plot_growth_prediction(cur_col)
                    plt.savefig(cur_col + '.png', dpi=200, bbox_inches='tight')
                    plt.clf()

        df = pd.DataFrame(growth_rate_data, columns=['well', 'growth_rate', 'max_time', 'max_index'])
        df['doubling_time'] = np.log(2)/df['growth_rate']

        return df