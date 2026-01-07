import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
import scipy.odr as odr
from scipy.differentiate import derivative

class Fit_Functions_L2():
    # the reason for so many functions is because I don't know how to rearange the arguments of functions
    # which is neccessary for derivative and ODR
    # lambda functions may work but weren't tried yet because of time constraints
    @staticmethod
    def fit_func_om(omega, params:list):
        a, omega_0, delta, B, C = params[1], params[2], params[3], params[4], params[5]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega + C)) + np.square(2 * delta * (omega + C))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_a(a, params:list):
        omega, omega_0, delta, B, C = params[0], params[2], params[3], params[4], params[5]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega + C)) + np.square(2 * delta * (omega + C))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_om0(omega_0, params:list):
        omega, a, delta, B, C = params[0], params[1], params[3], params[4], params[5]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega + C)) + np.square(2 * delta * (omega + C))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_del(delta, params:list):
        omega, a, omega_0, B, C = params[0], params[1], params[2], params[4], params[5]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega + C)) + np.square(2 * delta * (omega + C))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_B(B, params:list):
        omega, a, omega_0, delta, C = params[0], params[1], params[2], params[3], params[5]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega + C)) + np.square(2 * delta * (omega + C))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_C(C, params:list):
        omega, a, omega_0, delta, B = params[0], params[1], params[2], params[3], params[4]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega + C)) + np.square(2 * delta * (omega + C))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_odr_with_shift(params:list, omega):
        a, omega_0, delta, B, C = params[0], params[1], params[2], params[3], params[4]
        return a / np.sqrt(np.square(np.square(omega_0) - np.square(omega + C)) + np.square(2 * delta * (omega + C))) + B # = A(\omega)
    

    # The next functions are without the horizontal shift
    @staticmethod
    def fit_func_om_wo(omega, params:list):
        a, omega_0, delta, B = params[1], params[2], params[3], params[4]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega)) + np.square(2 * delta * (omega))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_a_wo(a, params:list):
        omega, omega_0, delta, B = params[0], params[2], params[3], params[4]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega)) + np.square(2 * delta * (omega))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_om0_wo(omega_0, params:list):
        omega, a, delta, B = params[0], params[1], params[3], params[4]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega)) + np.square(2 * delta * (omega))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_del_wo(delta, params:list):
        omega, a, omega_0, B = params[0], params[1], params[2], params[4]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega)) + np.square(2 * delta * (omega))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_B_wo(B, params:list):
        omega, a, omega_0, delta = params[0], params[1], params[2], params[3]
        res = a / np.sqrt(np.square(np.square(omega_0) - np.square(omega)) + np.square(2 * delta * (omega))) + B # = A(\omega)
        return np.array(res)

    @staticmethod
    def fit_func_odr_wo_shift(params:list, omega):
        a, omega_0, delta, B = params[0], params[1], params[2], params[3]
        return a / np.sqrt(np.square(np.square(omega_0) - np.square(omega)) + np.square(2 * delta * (omega))) + B # = A(\omega)


class evaluation():

    @staticmethod
    def u_A(params, u_params, horizontal_shift):
        # gaussian uncertainty propagation for amplitude
        result = 0
        if horizontal_shift == True:
            res = derivative(Fit_Functions_L2.fit_func_om, params[0], args = [params])
            result += np.square(res.df[0] * u_params[0])

            res = derivative(Fit_Functions_L2.fit_func_a, params[1], args = [params])
            result += np.square(res.df[0] * u_params[1])
            
            res = derivative(Fit_Functions_L2.fit_func_om0, params[2], args = [params])
            result += np.square(res.df[0] * u_params[2])
            
            res = derivative(Fit_Functions_L2.fit_func_del, params[3], args = [params])
            result += np.square(res.df[0] * u_params[3])

            res = derivative(Fit_Functions_L2.fit_func_B, params[4], args = [params])
            result += np.square(res.df[0] * u_params[4])
            
            res = derivative(Fit_Functions_L2.fit_func_C, params[5], args = [params])
            result += np.square(res.df[0] * u_params[5])
            
            return np.sqrt(result)
    
        res = derivative(Fit_Functions_L2.fit_func_om_wo, params[0], args = [params])
        result += np.square(res.df[0] * u_params[0])

        res = derivative(Fit_Functions_L2.fit_func_a_wo, params[1], args = [params])
        result += np.square(res.df[0] * u_params[1])
        
        res = derivative(Fit_Functions_L2.fit_func_om0_wo, params[2], args = [params])
        result += np.square(res.df[0] * u_params[2])
        
        res = derivative(Fit_Functions_L2.fit_func_del_wo, params[3], args = [params])
        result += np.square(res.df[0] * u_params[3])

        res = derivative(Fit_Functions_L2.fit_func_B_wo, params[4], args = [params])
        result += np.square(res.df[0] * u_params[4])
        
        return np.sqrt(result)
            

    @staticmethod
    def eval_L2(frequency:list, amplitude:list, axs, uncertainty_frequency, uncertainty_amplitude, beta0, horizontal_shift, axs_hist, test_phase = False):
        match horizontal_shift: # decide which fit function to use for ODR
            case True:
                func = Fit_Functions_L2.fit_func_odr_with_shift
            case False:
                func = Fit_Functions_L2.fit_func_odr_wo_shift
        odr_model = odr.Model(func) # create ODR model functions
        # load data and uncertainty into ODR object
        data = odr.RealData(frequency, amplitude, sx = uncertainty_frequency, sy = uncertainty_amplitude)
        odr_fit = odr.ODR(data,odr_model, beta0 = beta0, maxit = 10000) # Fit Data
        output = odr_fit.run() # run fit

        if test_phase == True:
            output.pprint() # print results if in test phase

        u_omega_0 = output.sd_beta[1]

        # create fit plot 
        x = np.linspace(min(frequency), max(frequency), 10000)
        y = [func(output.beta, i) for i in x]
        axs.plot(x,y, label = f"ODR Fit", c = "cyan")
    
        # find the resonance frequency by finding the peak of the fit function
        peaks, props = find_peaks(y)
        resonance_frequency = x[peaks[0]]
        # the uncertainty provided by find peaks cannot be used as the uncertainty for res freq because
        # it only describes the deviation from the ideal fit graph

        # MONTE CARLO for resonance frequency (find peaks) and cut off frequencies (least squares)
        # The algorithm performs the same calculations as the program does usually but the fit parameters
        # are randomized for each run
        runs = 1000
        result = []
        result_0 = []
        result_1 = []
        failures = 0

        def monte_carlo():
            arg = np.random.multivariate_normal(output.beta, output.cov_beta)
            x = np.linspace(min(frequency), max(frequency), 10000)
            y = [func(arg, i) for i in x]
            peaks, props = find_peaks(y)
            x2 = x[peaks[0]]
            result.append(x2)

            amplitude_res_freq = func(arg, x2)
            amplitude_cut_off_freq = 0.707 * amplitude_res_freq
            function_ls = lambda omega: func(arg, omega) - amplitude_cut_off_freq
            x0 = least_squares(function_ls, x0 = 1, bounds = (0, resonance_frequency))["x"][0]
            x1 = least_squares(function_ls, x0 = resonance_frequency + 2, bounds = (resonance_frequency, np.inf))["x"][0]
            result_0.append(x0)
            result_1.append(x1)

        for _ in range(runs):
            try:
                monte_carlo()
            except:
                failures += 1

        while failures != 0:
            for _ in range(failures):
                try:
                    monte_carlo()
                    failures -= 1
                except:
                    failures += 1

        # plot the range of the results as histograms
        axs_hist[0].hist(result, label = "Resonanzfrequenz")
        axs_hist[1].hist(result_0, label = "linke Grenzfrequenz")
        axs_hist[2].hist(result_1, label = "rechte Grenzfrequenz")

        # take the standard deviation of the results
        u_res_freq = np.std(result)
        u_cut_freq_0 = np.std(result_0)
        u_cut_freq_1 = np.std(result_1)


        # uncertainty of the amplitudes
        u_A_params = np.concatenate(([resonance_frequency],output.beta))
        u_A_unc_params = np.concatenate(([u_res_freq], output.sd_beta))

        amplitude_res_freq = func(output.beta, resonance_frequency)
        u_amplitude_res_freq = evaluation.u_A(u_A_params, u_A_unc_params, horizontal_shift)


        if test_phase == False: # plot resonance frequency with uncertainty
            axs.errorbar(resonance_frequency, amplitude_res_freq, xerr = u_res_freq, yerr = u_amplitude_res_freq, label = r"Resonanzfrequenz $\omega_0$",c = "blue", fmt = "o", capsize = 3)

        rel_u_f_res_freq = 100 * u_res_freq / np.abs(resonance_frequency)
        print(rf"The resonance frequency is: {resonance_frequency} {u'\u00b1'} {u_res_freq} with rel u. {rel_u_f_res_freq}")

        # calculate the cut off amplitudes
        amplitude_cut_off_freq = 0.707 * amplitude_res_freq
        u_amplitude_cut_off_freq = 0.707 * u_amplitude_res_freq
        rel_u_amp_res_freq = 100 * u_amplitude_res_freq / np.abs(amplitude_res_freq)
        rel_u_amp_cut_freq = 100 * u_amplitude_cut_off_freq / np.abs(amplitude_cut_off_freq)
        print(rf"Amplitude at res freq: {amplitude_res_freq} {u'\u00b1'} {u_amplitude_res_freq} with rel u. {rel_u_amp_res_freq}")
        print(rf"Amplitude at cut off freq: {amplitude_cut_off_freq} {u'\u00b1'} {u_amplitude_cut_off_freq} with rel u. {rel_u_amp_cut_freq}")
        
        # find the cut off frequencies by reducing the graph by the amplitude at which the frequencies are
        # and finding the roots
        function_ls = lambda omega: func(output.beta, omega) - amplitude_cut_off_freq
        cut_off_freq_0 = least_squares(function_ls, x0 = 1, bounds = (0, resonance_frequency))["x"][0]
        cut_off_freq_1 = least_squares(function_ls, x0 = resonance_frequency + 2, bounds = (resonance_frequency, np.inf))["x"][0]

        rel_u_cut_off_freq_0 = 100 * u_cut_freq_0 / np.positive(cut_off_freq_0)
        rel_u_cut_off_freq_1 = 100 * u_cut_freq_1 / np.positive(cut_off_freq_1)

        print(f"The left cut off frequencies are: {cut_off_freq_0} {u'\u00b1'} {u_cut_freq_0} with rel u. {rel_u_cut_off_freq_0}")
        print(f"The right cut off frequencies are: {cut_off_freq_1} {u'\u00b1'} {u_cut_freq_1} with rel u. {rel_u_cut_off_freq_1}")

        if test_phase == False: # plot cut off frequencies with uncertainty
            axs.errorbar(cut_off_freq_0, amplitude_cut_off_freq, xerr = u_cut_freq_0, yerr = u_amplitude_cut_off_freq, label = r"Linke Grenzfrequenz $f_{g,1}$", c = "green", fmt = "o", capsize = 3)
            axs.errorbar(cut_off_freq_1, amplitude_cut_off_freq, xerr = u_cut_freq_1, yerr = u_amplitude_cut_off_freq, label = r"Rechte Grenzfrequenz $f_{g,2}$", c = "purple", fmt = "o", capsize = 3)

        # calculate bandwidth
        bandwidth = cut_off_freq_1 - cut_off_freq_0
        u_bandwidth = np.sqrt(np.square(u_cut_freq_0) + np.square(u_cut_freq_1))
        rel_u_bandwidth = 100 * u_bandwidth / np.abs(bandwidth)
        print(rf"The bandwidth is equal to {bandwidth} {u'\u00b1'} {u_bandwidth} with rel u. {rel_u_bandwidth}")

        # calculate quality factor
        quality_factor = resonance_frequency / bandwidth
        u_quality_factor = np.sqrt(np.square(u_res_freq / bandwidth) + np.square(- resonance_frequency * u_bandwidth / np.square(bandwidth)))
        rel_u_qf = 100 * u_quality_factor / np.abs(quality_factor)
        print(rf"The quality factor is equal to {quality_factor} {u'\u00b1'} {u_quality_factor} with rel u. {rel_u_qf}")

        # calculate decay rate
        decay_rate = resonance_frequency / (2 * quality_factor)
        print(f"The decay rate is equal to: {decay_rate}")

if __name__ == "__main__":
    print("Wrong script dummy :)") # this happened way to often