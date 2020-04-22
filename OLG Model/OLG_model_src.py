import numpy as np
import pandas as pd
import altair as alt
from statsmodels.tsa.api import Holt


class Parameters:
    def __init__(self, tau, init_infected, fi, theta, countries, critical_condition_rate, recovery_rate,
                 critical_condition_time, recovery_time):
        self.tau = tau
        self.init_infected = init_infected
        self.fi = fi  # proportion of infectives are never diagnosed
        self.theta = theta  # diagnosis daily rate

        self.countries = countries
        self.critical_condition_rate = critical_condition_rate
        self.recovery_rate = recovery_rate
        self.critical_condition_time = critical_condition_time
        self.recovery_time = recovery_time

class OLG:
    """
    calc_asymptomatic start from first case
    exposed are the asymptomatic_infected with lag not only infected
    asymptomatic_infected eq 10 does not always grow but uses two diferent R0

    fi  # proportion of infectives are never diagnosed
    theta = theta  # diagnosis daily rate

    """

    def __init__(self, df, p: Parameters, jh_hubei, stringency, have_serious_data=True):
        self.detected = []
        self.r_adj = np.array([])
        self.r_values = np.array([])
        self.r0d = np.array([])
        self.asymptomatic_infected = []
        self.df = pd.DataFrame()
        self.df_tmp = pd.DataFrame()
        self.tmp = None
        self.have_serious_data = have_serious_data
        self.iter_countries(df, p, jh_hubei, stringency)
        self.r_hubei = None
        self.r_predicted = []
        self.day_0 = None

    @staticmethod
    def next_gen(r0, tau, c0, ct):
        r0d = r0 / tau
        return r0d * (ct - c0) + ct

    @staticmethod
    def true_a(fi, theta, d, d_prev, d_delta_ma):
        delta_detected = (d - d_prev)
        prev_asymptomatic_infected = (1 / (1 - fi)) * ((d_delta_ma) / theta + d_prev)
        return prev_asymptomatic_infected

    @staticmethod
    def crystal_ball_regression(r_prev, hubei_prev, hubei_prev_t2, s_prev_t7):
        crystal_ball_coef = {'intercept': 1.462, 'r_prev': 0.915, 'hubei_prev': 0.05, 'hubei_prev_t2': 0.058, 's_prev_t7': 0.0152}

        ln_r = crystal_ball_coef.get('intercept') + crystal_ball_coef.get('r_prev') * np.log(1+r_prev)\
                                                   + crystal_ball_coef.get('hubei_prev') * np.log(1+hubei_prev)\
                                                   + crystal_ball_coef.get('hubei_prev_t2') * np.log(1+hubei_prev_t2)\
                                                   - crystal_ball_coef.get('s_prev_t7') * s_prev_t7
        return np.exp(ln_r) - 1

    def iter_countries(self, df, p, jh_hubei, stringency):

        self.process(init_infected=250, detected=jh_hubei)
        self.calc_r(tau=p.tau, init_infected=250)
        self.r_hubei = self.r_adj
        r_hubei = self.r_adj
        for country in p.countries:
            self.df_tmp = df[df['country'] == country].copy()
            self.process(init_infected=p.init_infected)
            self.calc_r(tau=p.tau, init_infected=p.init_infected)
            if country == 'israel':
                self.predict(country, p.tau, r_hubei, stringency)
                self.predict_next_gen(tau=p.tau)
            self.calc_asymptomatic(fi=p.fi, theta=p.theta, init_infected=p.init_infected)
            self.write(stringency, tau=p.tau, critical_condition_rate=p.critical_condition_rate,
                       recovery_rate=p.recovery_rate, critical_condition_time=p.critical_condition_time,
                       recovery_time=p.recovery_time)


    def process(self, init_infected, detected=None):
        if detected is None:
            detected = self.df_tmp['total_cases'].values

        day_0 = np.argmax(detected >= init_infected)
        self.day_0 = day_0
        detected = detected[day_0:]
        self.detected = [detected[0]]
        for t in range(1, len(detected)):
            self.detected.append(max(detected[t - 1] + 1, detected[t]))

        if self.df_tmp is not None:
            self.df_tmp = self.df_tmp[day_0:]

    def calc_r(self, tau, init_infected):
        epsilon = 1e-06
        detected = self.detected
        r_values = np.array([(detected[0] / (init_infected + epsilon) - 1) * tau])
        for t in range(1, len(detected)):
            if t <= tau:
                r_value = (detected[t] / (detected[t - 1] + epsilon) - 1) * tau
            elif t > tau:
                r_value = (detected[t] / (detected[t - 1] - detected[t - tau] + detected[t - tau - 1] + epsilon) - 1) * tau
            r_values = np.append(r_values, max(r_value, 0))
            r_adj = np.convolve(r_values, np.ones(int(tau,)) / int(tau), mode='full')[:len(detected)]
            r_adj = np.clip(r_adj, 0, 100)
        self.r_values, self.r_adj, self.r0d = r_values, r_adj, r_adj

    def predict(self, country, tau, r_hubei, stringency):
        forcast_cnt = len(stringency)
        if country == 'israel':
            # fill hubei
            if 0 < forcast_cnt + len(self.r_adj) - len(r_hubei): ## TODO Shold use time series
                r_hubei = r_hubei.append(r_hubei[-1] * (forcast_cnt - len(r_hubei)))
            # StringencyIndex
            self.df_tmp['StringencyIndex'].fillna(method='ffill', inplace=True)
            tmp = self.df_tmp.copy()
            # normalize to day_0 and then shift forward 7 days so dont need to lag in regression
            tmp['StringencyIndex'].shift(periods= -(self.day_0 - 7)).fillna(method='ffill', inplace=True)
            cur_stringency = tmp['StringencyIndex'].values
            stringency = stringency['StringencyIndex'].values
            stringency = np.append(cur_stringency, stringency)
            self.r_predicted = [0]
            for t in range(1, len(self.r0d) + forcast_cnt+7):
                if t <= 2:
                    projected_r = self.r0d[t]
                else:
                    projected_r = self.crystal_ball_regression(self.r0d[t-1], r_hubei[t-1], r_hubei[t-2], stringency[min(t, len(stringency)-1)])
                if t >= len(self.r0d):
                    self.r0d = np.append(self.r0d, projected_r)
                self.r_predicted = np.append(self.r_predicted, projected_r)
        else:
            holt_model = Holt(self.r_adj[-tau:], exponential=True).fit(smoothing_level=0.1, smoothing_slope=0.9)
            self.r0d = np.append(self.r_adj, holt_model.forecast(forcast_cnt + 1))

        self.r0d = np.clip(self.r0d, 0, 100)

    def predict_next_gen(self, tau):
        t = len(self.detected)
        next_gen = self.detected[-1]
        c0 = self.detected[t - tau] if t - tau >= 0 else 0

        while t <= len(self.r0d) - 1:
            next_gen = self.next_gen(r0=self.r0d[t], tau=tau, c0=c0, ct=next_gen)
            self.detected.append(next_gen)
            t += 1

    def calc_asymptomatic(self, fi, theta, init_infected):
        detected_deltas = [self.detected[0]]

        for i in range(1, len(self.detected)):
            delta = self.detected[i] - self.detected[i-1]
            detected_deltas.append(delta)

        detected_deltas_ma = np.convolve(detected_deltas, np.ones(int(4, )) / int(4), mode='full')[:len(detected_deltas)]
        asymptomatic_infected = [self.true_a(fi=fi, theta=theta, d=self.detected[0], d_prev=init_infected, d_delta_ma= detected_deltas_ma[0])]

        for t in range(1, len(self.detected)):
            prev_asymptomatic_infected = self.true_a(fi=fi, theta=theta, d=self.detected[t],  d_prev=self.detected[t - 1], d_delta_ma=detected_deltas_ma[i])
            asymptomatic_infected.append(prev_asymptomatic_infected)
        self.asymptomatic_infected = asymptomatic_infected

    def calc_critical_condition(self, df, critical_condition_time, recovery_time, critical_condition_rate):
        # calc critical rate
        df['true_critical_rate'] = df['serious_critical'] / (
                    df['total_cases'].shift(critical_condition_time) - df['total_cases'].shift(
                critical_condition_time + recovery_time))
        critical_rates = df['serious_critical'] / (
                    df['total_cases'].shift(critical_condition_time) - df['total_cases'].shift(
                critical_condition_time + recovery_time))
        last_critical_rate = critical_rates.dropna().iloc[-7:].mean()

        # critical condition - currently using parameter
        df['Critical_condition'] = (df['total_cases'].shift(critical_condition_time)
                                    - df['total_cases'].shift(critical_condition_time + recovery_time + 1)) * critical_condition_rate

        return df['Critical_condition']

    def write(self, stringency, tau, critical_condition_rate, recovery_rate, critical_condition_time, recovery_time):
        if self.have_serious_data==False:
            self.df_tmp['serious_critical'] = None
            self.df_tmp['new_cases'] = self.df_tmp['total_cases'] - self.df_tmp['total_cases'].shift(1)
            self.df_tmp['activecases'] = None
            self.df_tmp['total_deaths'] = None
            self.df_tmp['new_deaths'] = None

        df = self.df_tmp[['date', 'country', 'StringencyIndex', 'serious_critical', 'new_cases', 'activecases','new_deaths', 'total_deaths']].reset_index(drop=True).copy()
        df['r_values'] = self.r_values
        # pad df for predictions
        forcast_cnt = len(self.detected) - len(self.r_adj)
        if forcast_cnt > 0:
            predict_date = df['date'].max() + pd.to_timedelta(1, unit="D")
            prediction_dates = pd.date_range(start=predict_date.strftime('%Y-%m-%d'), periods=forcast_cnt)
            predicted = pd.DataFrame({'date': prediction_dates})
            predicted.loc[:forcast_cnt - 8, 'StringencyIndex'] = stringency['StringencyIndex'].values
            df = df.append(predicted, ignore_index=True)

        df['total_cases'] = self.detected
        df['R'] = self.r0d

        df['infected'] = self.asymptomatic_infected
        df['exposed'] = df['infected'].shift(periods=-tau)
        df['country'].fillna(method='ffill', inplace=True)
        df['corona_days'] = pd.Series(range(1, len(df) + 1))
        df['prediction_ind'] = np.where(df['corona_days'] <= len(self.r_adj), 0, 1)
        df['Currently Infected'] = np.where(df['corona_days'] <= (critical_condition_time + recovery_time),
                                            df['total_cases'],
                                            df['total_cases'] - df['total_cases'].shift(periods=(critical_condition_time + 6+recovery_time)))

        df['Doubling Time'] = np.log(2) / np.log(1 + df['R'] / tau)

        df['dI'] = df['total_cases'] - df['total_cases'].shift(1)
        df['dA'] = df['infected'] - df['infected'].shift(1)
        df['dE'] = df['exposed'] - df['exposed'].shift(1)

        df['Critical_condition'] = self.calc_critical_condition(df, critical_condition_time, recovery_time, critical_condition_rate)
        df['Recovery_Critical'] = df['dI'].shift(recovery_time + critical_condition_time) * critical_condition_rate * recovery_rate
        df['Mortality_Critical'] = df['dI'].shift(recovery_time + critical_condition_time) * critical_condition_rate * (1-recovery_rate)
        df['Recovery_Critical'] = df['Recovery_Critical'].apply(lambda x: max(x, 0)).fillna(0).astype(int)
        df['Mortality_Critical'] = df['Mortality_Critical'].apply(lambda x: max(x, 0)).fillna(0).astype(int)

        df['Total_Mortality'] = df['Mortality_Critical'].cumsum()
        df['Total_Critical_Recovery'] = df['Recovery_Critical'].cumsum()

        # fill with obsereved values

        df[['Critical_condition', 'Currently Infected', 'total_cases', 'exposed', 'Recovery_Critical', 'Mortality_Critical']] = df[['Critical_condition', 'Currently Infected', 'total_cases', 'exposed', 'Recovery_Critical', 'Mortality_Critical']].round(0)
        if self.have_serious_data:
            df = df.rename(columns={'total_cases': 'Total Detected',
                                'infected': 'Total Infected Predicted',
                                'exposed': 'Total Exposed Predicted',
                                'Total_Mortality': 'Total Deaths Predicted',
                                'total_deaths':    'Total Deaths Actual',
                                'dI': 'New Detected Predicted',
                                'new_cases':'New Detected Actual',
                                'dA': 'New Infected Predicted',
                                'dE': 'New Exposed Predicted',
                                'Mortality_Critical': 'Daily Deaths Predicted',
                                'new_deaths':         'Daily Deaths Actual',
                                'Critical_condition' : 'Daily Critical Predicted',
                                'serious_critical':    'Daily Critical Actual',
                                'Recovery_Critical': 'Daily Recovery Predicted',
                                'Currently Infected': 'Currently Active Detected Predicted',
                                'activecases': 'Currently Active Detected Actual',
                                'true_critical_rate': 'Daily Critical Rate Actual',
                                'r_hubei': 'R China-Hubei Actual',
                                'r_predicted': 'R Predicted'
                                         })

        df['r_hubei'] = self.r_hubei[:df.shape[0]]
        if df.loc[0, 'country'] == 'israel':
            df['r_predicted'] = self.r_predicted[:df.shape[0]]
        self.df = pd.concat([self.df, df])


def plot_data(df, countries, var_in_multi_line='Total Detected'):

    country_count = df['country'].nunique()

    if country_count == len(countries):
        plot_df = df.query('prediction_ind==0').melt(id_vars=['corona_days'], value_vars=var_in_multi_line)
        plot_df_predict = df.query('prediction_ind==1').melt(id_vars=['corona_days'], value_vars=var_in_multi_line)

    else:
        plot_df = df.query('prediction_ind==0').pivot(index='corona_days', columns='Country',
                                                           values=var_in_multi_line).reset_index().melt(
            id_vars=['corona_days'],
            value_vars=countries)
        plot_df_predict = df.query('prediction_ind==1').pivot(index='corona_days', columns='Country',
                                                                   values=var_in_multi_line).reset_index().melt(
            id_vars=['corona_days'],
            value_vars=countries)

    plot_df.dropna(inplace=True)
    plot_df_predict.dropna(inplace=True)
    plot_df['value'] = plot_df['value']
    plot_df_predict['value'] = plot_df_predict['value']

    color_group = 'variable' if country_count == 1 else 'country'

    # The basic line
    line = alt.Chart(plot_df).mark_line(interpolate='basis').encode(
        x='corona_days:Q',
        y='value',
        color=color_group
    )

    line2 = alt.Chart(plot_df_predict).mark_line(interpolate='basis', strokeDash=[1, 1]).encode(
        x='corona_days:Q',
        y='value',
        color=color_group
    )

    return alt.layer(
        line, line2
    ).properties(
        width=600, height=300
    )

