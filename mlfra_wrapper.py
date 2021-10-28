import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy import optimize
from scipy.integrate import simps
import time
import sys

def SmoothFilter(waveform, fs, window_size, order_poly):
    #Return raw waveform if too short
    if len(waveform) < window_size:
        return waveform

    #Run a savolitz-folay filter and return the result
    waveform_smooth = savgol_filter(waveform, window_size, order_poly)
    return waveform_smooth

def find_systole(filtered_signal, fs):
    #Do peak detection to identify arterial systole
    beat_duration = 25
    peaks, _ = find_peaks(filtered_signal, distance=beat_duration, prominence=0.7)
    
    #Get the filtered value at e
    #ach peak
    peaks_y = [filtered_signal[i] for i in peaks]

    new_peaks = []
    new_peaks_y = []

    i_list = set()
    for i in range(11, len(peaks_y)):
        temp = peaks_y[i-9:i+1]
        temp.sort()
        criteria = temp[7]
        if peaks_y[i] >= 0.87 * criteria:
            i_list.add(i)

    for i in range(len(peaks_y)-11,-1,-1):
        temp = peaks_y[i+1:i+11]
        temp.sort()
        criteria = temp[7]
        if peaks_y[i] >= 0.87*criteria:
            i_list.add(i)
    
    ind_list = sorted(i_list)
    for ind in ind_list:
        new_peaks.append(peaks[ind])
        new_peaks_y.append(peaks_y[ind])

    return new_peaks, new_peaks_y

def find_max_local_peak(signal, vec, main_locality, left_locality=None, right_locality=None):
    if right_locality is None:
        right_locality = main_locality
    if left_locality is None:
        left_locality = main_locality
    vec = np.array(vec)
    signal = np.array(signal)
    res = []
    total_height = signal[0] - signal[-1]
    main_r = total_height * main_locality
    left_ind = 1
    for i in range(1, len(signal)):
        if signal[0] - signal[i] >= total_height * left_locality:
            left_ind = i
            break
    right_ind = len(vec)-2
    x_threshold = len(signal)*0.8
    for i in range(len(vec)-2, -1, -1):
        if signal[i] - signal[-1] >= total_height * right_locality or i < x_threshold:
            right_ind = i
            break
    for i in range(left_ind, right_ind+1):
        l_ind = i - 1
        while l_ind >= 1 and signal[l_ind-1] - signal[i] <= main_r:
            l_ind -= 1
        r_ind = i + 1
        while r_ind <= len(signal)-2 and signal[i] - signal[r_ind+1] <= main_r:
            r_ind += 1
        local_max_shifted = np.argmax(vec[l_ind:r_ind+1])
        if local_max_shifted + l_ind == i:
            res.append(i)
    return res

def find_min_local_peak(signal, vec, main_locality, left_locality=None, right_locality=None):
        return find_max_local_peak(signal, [-x for x in vec], main_locality, left_locality, right_locality)

def find_min_ind(signal):
    period = len(signal) // 3
    for i in range(period, len(signal)-period):
        if signal[i] == np.min(signal[i-period:i+period+1]):
            return i

def calculate_curve(signal_view, i, lat):
    class ComputeCurvature:
        def __init__(self):
            """ Initialize some variables """
            self.xc = 0  # X-coordinate of circle center
            self.yc = 0  # Y-coordinate of circle center
            self.r = 0   # Radius of the circle
            self.xx = np.array([])  # Data points
            self.yy = np.array([])  # Data points

        def calc_r(self, xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((self.xx-xc)**2 + (self.yy-yc)**2)

        def f(self, c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            ri = self.calc_r(*c)
            return ri - ri.mean()

        def df(self, c):
            """ Jacobian of f_2b
            The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
            xc, yc = c
            df_dc = np.empty((len(c), x.size))

            ri = self.calc_r(xc, yc)
            df_dc[0] = (xc - x)/ri                   # dR/dxc
            df_dc[1] = (yc - y)/ri                   # dR/dyc
            df_dc = df_dc - df_dc.mean(axis=1)[:, np.newaxis]
            return df_dc

        def fit(self, xx, yy):
            self.xx = xx
            self.yy = yy
            center_estimate = np.r_[np.mean(xx), np.mean(yy)]
            center = optimize.leastsq(self.f, center_estimate, Dfun=self.df, col_deriv=True)[0]

            self.xc, self.yc = center
            ri = self.calc_r(*center)
            self.r = ri.mean()

            return 1/self.r  # Return the curvature

    N = len(signal_view)

    if i < lat or i+lat > len(signal_view):
        return 0

    x = np.linspace(i-lat, i+lat, 2*lat,endpoint=False)
    y = np.array(signal_view[i-lat:i+lat])

    comp_curv = ComputeCurvature()
    return int(comp_curv.r)

def find_relative_dichrotic(signal_view, relative_diastol):
    #Calculate a window
    window = relative_diastol//4
    if window % 2 == 0:
        window += 1
    if window < 2:
        return 0

    #Use savgol to create 1st order and 2nd order deriviative
    diff_w = savgol_filter(signal_view[:relative_diastol], window_length = window, polyorder = 2, deriv = 1)
    diff2_w = savgol_filter(signal_view[:relative_diastol], window_length = window, polyorder = 2, deriv = 2)

    #Find local min peak in the 2nd order derivative
    peak_locality = 0.03
    low_peaks = find_min_local_peak(signal_view[0:relative_diastol], diff2_w, peak_locality, right_locality=peak_locality/6)
    if len(low_peaks) == 0:
        min_acc_ans = max(int(0.1 * relative_diastol), 1)
        max_acc_ans = max(relative_diastol - min_acc_ans + 1, min_acc_ans + 1)
        fallback_ans = min_acc_ans + np.argmax(diff2_w[min_acc_ans:max_acc_ans])
        #print("no low peaks!", fallback_ans)
        return fallback_ans

    #Get the last low peak 
    end = low_peaks[-1]
    eligible_peaks = find_max_local_peak(signal_view[0:relative_diastol], diff2_w, peak_locality)
    if len(eligible_peaks) == 0:
        min_acc_ans = min(max(int(0.1 * relative_diastol), 1), end - 1)
        fallback_ans = min_acc_ans + np.argmax(diff2_w[min_acc_ans:end])
        #print("no high peaks!", fallback_ans)
        return fallback_ans

    # Find plateau for each high peak
    abs_diff_w = [abs(x) for x in diff_w]
    plateau_map = {}
    black_listed_peaks = set()
    for i in range(len(eligible_peaks)):
        if i in black_listed_peaks:
            continue
        high_peak = eligible_peaks[i]
        next_low_peak = -1
        for x in low_peaks:
            if x > high_peak:
                next_low_peak = x
                break
        if next_low_peak == -1:
            black_listed_peaks.add(i)
            continue
        j = 1
        is_valid = True
        while i+j < len(eligible_peaks) and eligible_peaks[i+j] <= next_low_peak:
            # this high peak shares plateau with the peak at i+j
            if diff2_w[eligible_peaks[i+j]] > diff2_w[high_peak]:
                black_listed_peaks.add(i)
                is_valid = False
                break
            else:
                black_listed_peaks.add(i+j)
                j += 1
        if not is_valid:
            continue

        least_abs_diff_w = min(abs_diff_w[high_peak:next_low_peak])
        min_ind_old = np.argmin(signal_view[high_peak:next_low_peak+1])
        min_ind = find_min_ind(signal_view[high_peak:next_low_peak+1])

        local_minima_resilience = (next_low_peak-high_peak+1)*0.1
        if min_ind!= None and (min_ind <= local_minima_resilience or ((min_ind+high_peak) < next_low_peak-local_minima_resilience and diff_w[next_low_peak] > diff_w[min_ind])):
                best_plateau_ind = min_ind+high_peak
        else:

            end_of_peak_resilience = 0.85
            high_der2 = high_peak+np.argmax(diff2_w[high_peak:next_low_peak+1])

            low_der2 = high_peak+np.argmin(diff2_w[high_peak:next_low_peak+1])
            best_plateau_ind = high_der2

            champion_req_diff2 = (
                diff2_w[low_der2] +
                end_of_peak_resilience * (diff2_w[high_der2] - diff2_w[low_der2])
            )

            while (
                best_plateau_ind + 1 < next_low_peak and
                diff2_w[best_plateau_ind + 1] >= champion_req_diff2
            ):
                best_plateau_ind += 1

        """choosing the point that its firs tderivative is small"""
        for ind in range(best_plateau_ind,next_low_peak):
            if diff_w[ind] > -0.5:
                break
        best_plateau_ind = ind

        """ if curve at high peak is around zero and its slope is also zero, then check best pleatu ind, if curve is not near zero, then find the next low curve and then best pleatu ind after that"""
        max_curve = 0
        max_curve_ind = high_peak
        pre_curve = 0

        min_curve = float('inf')
        for s in range(high_peak,next_low_peak):
            curve = calculate_curve(signal_view, s, 30)
            if curve < pre_curve*0.99:
                max_curve_ind = s
                break
            pre_curve = curve

        plateau_map[high_peak] = {
            'best_plateau_ind':best_plateau_ind,
            'least_abs_diff_w':least_abs_diff_w,
            'next_low_peak':next_low_peak,
            'max_curve_ind': max_curve_ind
        }
    
    # Drop the peaks without a plateau
    eligible_peaks = [
        x for x in eligible_peaks
        if x in plateau_map
    ]
    if len(eligible_peaks) == 0:
        min_acc_ans = min(max(int(0.1 * relative_diastol), 1), end-1)
        fallback_ans = min_acc_ans + np.argmax(diff2_w[min_acc_ans:end])
        #print("no plateau!", fallback_ans)
        return fallback_ans

    ## filtering high peaks based on comparing with median of 3 top max high peaks in second derivative
    derivatives = [diff2_w[X] for X in eligible_peaks]
    derivatives.sort()
    if derivatives[-1] > 0.03:
        if len(derivatives) == 1:
            threshold = 0.03
        elif len(derivatives) > 1:
            threshold = 0.03
        eligible_peaks = [
            x for x in eligible_peaks
            if diff2_w[x] >= threshold
        ]

    # Drop the peaks that their plateau is not FLAT enough
    flat_resilience = 0.25
    max_eligible_abs_diff = max(
        flat_resilience,
        min(plateau_map[x]['least_abs_diff_w'] for x in eligible_peaks) * (1.0 + flat_resilience)
    )
    eligible_peaks = [
        x for x in eligible_peaks
        if plateau_map[x]['least_abs_diff_w'] <= max_eligible_abs_diff
    ]

    #Drop the peaks that their plateau is not WIDE enough
    width_resilience = 0.5
    min_eligible_width = max(plateau_map[x]['next_low_peak'] - x for x in eligible_peaks) * width_resilience
    eligible_peaks = [
        x for x in eligible_peaks
        if plateau_map[x]['next_low_peak'] - x >= min_eligible_width
    ]

    mid_score = []
    for x in eligible_peaks:
        val = max(x, relative_diastol - x)
        mid_score.append([val,x])

    mid_score.sort(key=lambda item:item[0])
    best_high_peak = mid_score[0][1]
    best_val = mid_score[0][0]

    if len(mid_score) > 1:
        dis1 = mid_score[-1][0]
        ind1 = mid_score[-1][1]
        dis2 = mid_score[-2][0]
        ind2 = mid_score[-2][1]
        if dis2 - dis1 < 3:
            if abs(diff_w[ind2]) < abs(diff_w[ind1]):
                best_high_peak = ind2
                best_val = dis2

    relative_dicro = (
        plateau_map[best_high_peak]['best_plateau_ind']
    )

    return relative_dicro

def find_diastolic_and_dichrotic(ind1, ind2, signal, Systole, Dicro_peak, End_diastole, fs):
    #Invert the signal and do peak detection between two indicies
    negate_signal = [-1*item for item in signal[ind1:ind2]]
    found = False
    min_peak, _ = find_peaks(negate_signal)

    #Parse the peaks output
    if len(min_peak) == 0:
        relative_diastol = np.argmin(signal[ind1:ind2])
        found = True
        #print("nodrat", ind1, ind2, relative_diastol, relative_diastol + ind1)
    else:
        t = min(signal[ind1:ind2])-signal[ind1]
        for x in reversed(min_peak):
            if signal[x+ind1]-signal[ind1] < t*0.95:
                relative_diastol = x
                found = True
                break

    if not found:
        relative_diastol = min_peak[-1]
        #print("No DBP found!", ind1, ind2)

    End_diastole.append(relative_diastol + ind1)

    #Find dicrotic notch
    relative_dicro = find_relative_dichrotic(signal[ind1:ind2], relative_diastol)
    Dicro_peak.append(relative_dicro + ind1)

class Beat:
    def __init__(self):
        self.Systolic_x = 0
        self.Systolic_y = 0

        self.Diastolic_x = 0
        self.Diastolic_y = 0

        self.Dicrotic_x = 0
        self.Dicrotic_y = 0

def create_beats(SYS, DIA, DCN, theSignal):
    index = {'Systolic':0, 'Diastolic':0, 'Dicrotic': 0}
    beats = []

    lastSys = None
    for i in range(0, len(SYS)):
        if(not lastSys == None):
            beats.append(Beat())
            setattr(beats[i - 1], "Systolic_x", lastSys)
            setattr(beats[i - 1], "Systolic_y", theSignal[lastSys])

            for j in range(len(DIA)):
                if(DIA[j] > lastSys and DIA[j] < SYS[i]):
                    setattr(beats[i - 1], "Diastolic_x", DIA[j])
                    setattr(beats[i - 1], "Diastolic_y", theSignal[DIA[j]])
                    break

            for j in range(len(DCN)):
                if(DCN[j] > lastSys and DCN[j] < SYS[i]):
                    setattr(beats[i - 1], "Dicrotic_x", DCN[j])
                    setattr(beats[i - 1], "Dicrotic_y", theSignal[DCN[j]])
                    break
        #update 
        lastSys = SYS[i]

    return beats

def PreProcess(waveform, fs):
    #Run the smoothing algorithm on raw waveform
    smoothedSignal = SmoothFilter(waveform, fs, 19, 2)

    #Find the systsolic peaks
    peaks_ind, peaks_y = find_systole(smoothedSignal, fs)
    Systole = []
    for i in range(len(peaks_ind)):
        Systole.append(peaks_ind[i])

    #Search between systolic and diastolic peaks for diacrotic notch and diastole
    End_diastole = []
    Dicrotic_peak = []
    for i in range(len(peaks_ind) - 1):
        find_diastolic_and_dichrotic(peaks_ind[i], peaks_ind[i+1], smoothedSignal, Systole, Dicrotic_peak, End_diastole, fs)

    return(Systole, End_diastole, Dicrotic_peak)

def Abp_features(beats, signal, fs):
    col_names = [
    'sys_pres',
        'dias_pres',
        'dic_pres',
        'map',
        'heart_rate',
        't_sys_rise',
        't_sys',
        't_dia',
        'ibi',
        'pulse_pres',
        'avg_sys_rise',
        'avg_sys',
        'avg_dia',
        'pp_area',
        'pp_area_nor',
        'sys_area',
        'sys_area_nor',
        'sys_rise_area',
        'sys_rise_area_nor',
        'sys_dec_area',
        'sys_dec_area_nor',
        'dec_area',
        'dec_area_nor',
        'dia_area',
        'dia_area_nor']

    data = []
    for i in range(1,len(beats)):
        temp = []
        systole_time = beats[i].Systolic_x
        dicrotic_time = beats[i].Dicrotic_x

        it = 0
        beat_ind_list = []
        for j in range(len(signal)):
            if j >= beats[i-1].Diastolic_x and j <= beats[i].Diastolic_x:
                beat_ind_list.append(j)

        ## start ind
        try:
            beat_start_ind = beat_ind_list[0] + 1
            beat_start_time = beat_start_ind / fs
        except:
            break

        ## end ind
        beat_end_ind = beat_ind_list[-1]
        beat_end_time = beat_end_ind / fs

        ## sys ind
        for j in range(beat_start_ind, beat_end_ind):
            if j > beats[i].Systolic_x:
                break
        sys_ind = j-1

        for j in range(beat_start_ind,beat_end_ind):
            if j > beats[i].Dicrotic_x:
                break
        dic_ind = j-1

        for j in range(beat_start_ind,beat_end_ind):
            if j > beats[i].Diastolic_x:
                break
        dia_ind = j-1

        try:
            # systolic pressure
            temp.append(beats[i].Systolic_y)

            # diastolic pressure
            temp.append(beats[i].Diastolic_y)

            # dicrotic pressure
            temp.append(beats[i].Dicrotic_y)

            # mean arterial pressure = map
            temp.append(np.mean(signal[beat_start_ind:beat_end_ind]))

            ## heart rate
            temp.append(60/(beat_end_time - beat_start_time))

            ## phase 3
            temp.append(systole_time - beat_start_time)

            ### t_sys	The duration of the systolic phase (Phase 1)
            temp.append(dicrotic_time - beat_start_time)

            ## t_dia	The duration of the diastolic phase (Phase 2)
            temp.append(beat_end_time - dicrotic_time)
                
            ## ibi	Interbeat interval	 
            temp.append(systole_time - beats[i-1].Systolic_x)

            ## pulse_pres	Pulse pressure	
            temp.append(beats[i].Systolic_y - beats[i].Diastolic_y)
                
            #avg_sys_rise	Average of the systolic rise portion of the waveform (Phase 3)	 
            temp.append(np.mean(signal[beat_start_ind:sys_ind]))
            # avg_sys	Average of the systolic portion of the waveform (Phase 1)	
            temp.append(np.mean(signal[beat_start_ind:dic_ind]))

            #avg_dia	Average of the of the diastolic portion of the waveform (Phase 2)	
            temp.append(np.mean(signal[dic_ind:beat_end_ind]))

            # pp_area	Area under the beat	" ## AUC under entire beat
            curve = signal[beat_start_ind:beat_end_ind]
            area = simps(curve)/100
            temp.append(area)
                
            #pp_area_nor	Area under the beat normalized by the number of samples	"                
            curve = signal[beat_start_ind:beat_end_ind]
            area = simps(curve)/100
            temp.append(area/len(curve))
                
            # sys_area	Area under the systolic phase of the beat (from start to the dicrotic notch)	"               
            curve = signal[beat_start_ind: dic_ind]
            area = simps(curve)/100
            temp.append(area)

            # sys_area_nor	Area under the systolic phase normalized by the number of samples	"                
            curve = signal[beat_start_ind:dic_ind+1]
            area = simps(curve)/100
            temp.append(area/len(curve))
                
            #sys_rise_area	Area from the start of the beat to the systolic maximum	"                
            curve = signal[beat_start_ind:sys_ind+1]
            area = simps(curve)/100
            temp.append(area)
                
            #sys_rise_area_nor	Area from the start of the beat to the systolic maximum normalized by the number of samples	"                
            curve = signal[beat_start_ind:sys_ind+1]
            area = simps(curve)/100
            temp.append(area/len(curve))
                
            # sys_dec_area	Area from the systolic maximum to the dicrotic notch	"                
            curve = signal[sys_ind:dic_ind+1]
            area = simps(curve)/100
            temp.append(area)
                
            #sys_dec_area_nor	Area from the systolic maximum to the dicrotic notch normalized by the number of samples	"                
            curve = signal[sys_ind:dic_ind+1]
            area = simps(curve)/100
            temp.append(area/len(curve))
                
            #dec_area	Area from the systolic maximum to the start of the next beat	"                
            curve = signal[sys_ind:beat_end_ind]
            area = simps(curve)/100
            temp.append(area)
                
            #dec_area_nor	Area from the systolic maximum to the start of the next beat normalized by the number of samples	"                
            curve = signal[sys_ind:beat_end_ind]
            area = simps(curve)/100
            temp.append(area/len(curve))
                
            #dia_area Area under the diastolic portion of the waveform 	"                
            curve = signal[dic_ind:beat_end_ind]
            area = simps(curve)/100
            temp.append(area)

            # dia_area_nor	Area under the diastolic portion of the waveform normalized by the number of samples	"                
            curve = signal[dic_ind:beat_end_ind]
            area = simps(curve)/100
            temp.append(area/len(curve))
        except Exception as err:
            #print(err)
            temp = []

        data.append(temp)

    features = pd.DataFrame(data, columns = col_names)
    return features

def calc_stat(feature_column, column_name):
    names = []
    sorted_column = sorted(feature_column)
    ans = []

    if len(feature_column) == 0:
        return names, ans

    names = [
    'min_{}'.format(column_name),
    'max_{}'.format(column_name),
    'mean_{}'.format(column_name),
    'median_{}'.format(column_name),
    'std_{}'.format(column_name)]

    ans = [
    sorted_column[0],
    sorted_column[-1],
    np.mean(feature_column),
    sorted_column[len(feature_column)//2],
    np.std(feature_column)]

    return names, ans

def Variability_features(current_features):
    cols = current_features.columns
    var_features_names = []
    var_features_vals = []
    for col in cols:
        feature_names, feature_vals = calc_stat(list(current_features[col].values),col)
        var_features_names += feature_names
        var_features_vals += feature_vals

    var_features_df = pd.DataFrame(columns = var_features_names)
    for i in range(len(var_features_names)):
        var_features_df[var_features_names[i]] = [var_features_vals[i]]

    return var_features_df

def MLFRA_Wrapper(theSignal, sampleFreq):
    #Only designed to work on 100 hz data
    fs = None
    if(not sampleFreq == 100): 
        raise Exception("Frequency can only be 100 Hz")
    else: 
        fs = 100.0

    #Process the signal into beats
    systole, diastole, dicroticNotch = PreProcess(theSignal, 100)
    theBeats = create_beats(systole, diastole, dicroticNotch, theSignal)

    #Extract ABP features
    abpFeatSet = Abp_features(theBeats, theSignal, fs)

    #Get stats of ABP features
    varABPFeat = Variability_features(abpFeatSet)

    try:
        #Get the features from processed array
        Features =['std_sys_pres', 'std_avg_sys_rise', 'median_dias_pres', 'std_pulse_pres', 'median_sys_rise_area', 'std_pp_area',
                   'median_pulse_pres', 'std_dic_pres', 'median_sys_area', 'std_avg_dia','median_t_sys', 'median_t_sys_rise', 'std_t_dia', 'std_dias_pres',
                   'std_t_sys', 'median_dia_area', 'std_sys_dec_area', 'median_ibi']
        X = varABPFeat[Features]

        #Predict fluid responsiveness
        joblib_model = joblib.load('./MLFRA_ABP_model.pkl')
        Y_pred = joblib_model.predict(X)
        probas_ = joblib_model.predict_proba(X)

        #Return the probabilty
        return(Y_pred[0], ((probas_[:,1][0])*100.0))

    except Exception as err:
        print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
        print(err)
        return(-1, None)

if(False):
    try:
        theInput = sys.argv[1]
        f = open(theInput)
        dataString = f.read().strip("\n")
        f.close()
        #dataString = input()

        theArr = []    
        for theI in dataString.split(','):
            theArr.append(float(theI))
       
        outRes = MLFRA_Wrapper(theArr, 100)
        print(str(outRes[0]) + "," + str(outRes[1]))
    except Exception as err:
        print("-2,-1")

