import numpy as np
import pickle, os, random, math
from networks_v0_2 import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats, signal
import matplotlib.pyplot as plt


######################################## Defining functions for filtering and results###################################
def sum_filt(sig_i, win_type='uniform', win_len=14, use_future_samples=True, plot_examp=False):
    if win_type == 'uniform':
        win = np.ones((win_len))  # window with size win_len
    elif win_type == 'half_gaussian':
        win = signal.gaussian(win_len*2, std=5)[win_len:] #taking the second half because it will be flipped in the
                                                            # convolve function.

    # padding the signal by repeating the first and last value
    sig_i_f = np.append(sig_i, np.zeros((sig_i.shape[0], win_len - 1)), axis=1)
    sig_i_f = np.append(np.zeros((sig_i.shape[0], win_len - 1)), sig_i_f,
                              axis=1)
    for sample_i in range(sig_i.shape[0]):
        sig_i_f[sample_i, :win_len - 1] = np.repeat(sig_i_f[sample_i, win_len - 1], win_len - 1)
        sig_i_f[sample_i, -win_len + 1:] = np.repeat(sig_i_f[sample_i, -win_len], win_len - 1)

    for sample_i in range(sig_i.shape[0]):
        sig_i_f[sample_i, :] = signal.convolve(sig_i_f[sample_i, :], win, mode='same')# use mode='same' if all the
                                                                                        # signal is use, it pads the
                                                                                        # signal with zeros.
    if use_future_samples: #sum up based on the next and previous 7 samples
        sig_i_f = sig_i_f[:, win_len-1: -win_len+1]
    else: #sum up based on the previous 14 samples
        sig_i_f = sig_i_f[:, win_len - int(win_len / 2): -win_len - int(win_len / 2) + 2]

    if plot_examp:
        plt.figure()
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.plot(np.arange(1, sig_i.shape[1] + 1), sig_i[i, :], 'b',
                     np.arange(1, sig_i_f.shape[1] + 1),
                     sig_i_f[i, :], 'g')
            plt.legend(['Actual', 'Actual_filtered'], fontsize='large')
            plt.xlabel('Day #', fontsize='large', weight='bold')
            plt.xticks(fontsize=12, rotation=0)
            plt.yticks(fontsize=12, rotation=0)
            plt.grid(True)
        plt.show
    return sig_i_f

def data_filter(data, county_list, population_dens):
    sel_counties = np.ones(shape=county_list.__len__(), dtype=np.bool)
    for count_i in range(data_all.shape[0]):
        if data_all[count_i, 0, 56] < population_dens: #56 is the population density
            sel_counties[count_i] = False
    return data[sel_counties, :, :], county_list[sel_counties]

def calculate_results(labels, labels_pred, mean_train, std_train):
    # de-normalize the data
    labels = (labels * std_train) + mean_train
    labels_pred = (labels_pred * std_train) + mean_train

    #Metrics
    results_temp = np.zeros((5))
    results_temp[0] = np.sqrt(
        mean_squared_error(labels, labels_pred))
    results_temp[1] = mean_absolute_error(labels,labels_pred)
    results_temp[2] = r2_score(labels,labels_pred)
    corrI = stats.pearsonr(labels, labels_pred)
    results_temp[3] = corrI[0]
    results_temp[4] = corrI[1]
    return results_temp



if __name__=='__main__':
    ######################################## General training parameters################################################
    test_date = 'd2020_08_01'
    num_lstm_layers = [1, 2, 3]
    num_lstm_states = [32, 64, 128, 192, 256, 320]
    SaveDirParent = './' + os.path.basename(__file__)[0:-3] + '_results/'
    if os.path.isdir(SaveDirParent) == False:
        os.mkdir(SaveDirParent)
    batch_size = 32
    num_epochs = 200
    num_epochs_adapt = 50
    base_learning_rate = 0.001
    decayRate=base_learning_rate/num_epochs
    momentumValue = 0.9
    dropout_prob = 0.5
    lambda_loss_amount = 0.005
    num_classes = 1
    adapt_train = 7 #number of days to start each fine-tuning
    val_perc = 0.2 #20% pecent of counties from the training data to be splitted for validation set.

    ######################################## Loading and preparing data#################################################
    with open('./data_v0_6.pkl', 'rb') as data_combined:
        [data_all, county_list, features_all, dates_all] = pickle.load(data_combined)
        #data_all: an numpy array to of the daily data for each county in the US: County x features x date
        #county_list: ordered US counties representing counties in data_all. Counties were discarded if one of
            # the features for this county are not available for all the days.
        #features_all: a list of all the features in data_all.
        #dates_all: dates list corresponds to the date axis in data_all.
    start_date = 'd2020_02_15'
    end_date= 'd2021_01_22'

    #finding new recovered cases as feature and add it to features
    data_all = np.append(data_all, np.zeros((data_all.shape[0], data_all.shape[1], 1)), axis=2)
    for i in range(data_all.shape[0]):
        data_all[i,1:,-1] = data_all[i, 1:, 27]-data_all[i, :-1, 27] #feature 27 is total recoverd cases
        data_all[i, data_all[i, :, -1] < 0, -1] = 0 #Some daily revovered are negative! so zero them out
        data_all[i, data_all[i, :, 23] < 0, 23] = 0 #Some daily cases are negative! so zero them out, only during the first week of the pandamic
        data_all[i, data_all[i, :, 25] < 0, 25] = 0 #Some daily deaths are negative! so zero them out
    features_all.append('newrecovered')


    [data_all, county_list] = data_filter(data_all, county_list, population_dens=150) #filtering counties based on
                                                                                        # the population density


    ######################################## Defining the target and selecting features#################################
    #Data labels
    prediction_day = 14
    predicted_attributes = [23]
    new_labels = ['nday_accum_cases']
    print('The predicted attributes accumulated for coming day #' + str(prediction_day) + ' are:')
    print(new_labels[0])
    #Adding a new feature with is current new labels
    data_all = np.append(data_all, np.zeros((data_all.shape[0], data_all.shape[1], 1)), axis=2)
    data_all[:, :, -1] = sum_filt(data_all[:, :, predicted_attributes[0]], win_type='uniform', win_len=prediction_day,
                                  use_future_samples=False, plot_examp=False)
    features_all.extend([new_labels[0]])
    labels_all = data_all[:, prediction_day:, [-1]] #using the future new labels that were added as the last feature
    data_all = data_all[:, :-prediction_day, :] #no labels for the last days used for prediction
    predicted_attributes = [data_all.shape[2]-1] #now the predicted attribute is the last feature

    #Selecting some features
    sel_featuers = [0, 36, 23, 25, 57, 49, 50, 56, 58] #'cars_mobility_apple', 'immunepct','newcases', 'newdeaths', 'newrecovered',
                                                    #'fips_car_dens', 'fips_pop_ratio', 'fips_pop_density', 'nday_accum_cases'
    perc_teen = np.expand_dims(np.sum(data_all[:, :, [38, 39]], axis=2), axis=2)/ data_all[:, :, [37]]
    perc_adult= np.expand_dims(np.sum(data_all[:, :, [40, 41]], axis=2), axis=2)/ data_all[:, :, [37]]
    perc_ret = np.expand_dims(np.sum(data_all[:, :, [42, 43, 44]], axis=2), axis=2)/ data_all[:, :, [37]]
    #perc_infected = data_all[:, :, [22]]/ data_all[:, :, [37]] #'adjcumcases'/'fips_occupants'
    data_all = np.append(data_all[:, :, sel_featuers], perc_teen, axis=2)
    data_all = np.append(data_all, perc_adult, axis=2)
    data_all = np.append(data_all, perc_ret, axis=2)
    #data_all = np.append(data_all, perc_infected, axis=2)
    features_all = ['cars_mobility_apple', 'immunepct', 'newcases', 'newdeaths', 'newrecovered', 'fips_car_dens',
                    'fips_pop_ratio', 'fips_pop_density', 'nday_accum_cases', 'percentage_teen', 'percentage_adult',
                    'percentage_retired'] #, 'percentage_infected'
    predicted_attributes = [8] #now the predicted attribute is feature 7
    del perc_teen, perc_adult, perc_ret


    ######################################## Spitting the data into training and testing################################
    print('Using the data before ' + str(test_date) + ' for training, and the rest for testing.')
    test_date_ind = np.where(dates_all==test_date)[0][0]
    data_train = np.copy(data_all[:, :test_date_ind, :])
    labels_train = np.copy(labels_all[:, :test_date_ind, :])
    num_samples_train = data_train.shape[0]
    data_test = np.copy(data_all[:, test_date_ind:, :])
    labels_test = np.copy(labels_all[:, test_date_ind:, :])
    num_samples_test = data_test.shape[0]


    ############################################## Splitting training data, and normalization###########################
    # Creating training and validation splits (80%,20%) from training folds #
    random.seed(100)
    train_inds = random.sample(range(0, num_samples_train), num_samples_train)
    val_county_inds = train_inds[0:math.ceil(num_samples_train * val_perc)]
    train_county_inds = train_inds[math.floor(num_samples_train * val_perc) + 1:num_samples_train]
    random.seed()
    data_val = data_train[val_county_inds]
    labels_val = labels_train[val_county_inds]
    num_samples_val = data_val.shape[0]  # number of samples for training
    data_train = data_train[train_county_inds]
    labels_train = labels_train[train_county_inds]
    num_samples_train = data_train.shape[0]  # number of samples for training

    #Normalization
    #data_train = np.swapaxes(data_train, 1, 2)
    data_train_temp = np.reshape(data_train, (data_train.shape[0]*data_train.shape[1], data_train.shape[2]))
    mean_train = np.expand_dims(np.nanmean(data_train_temp, axis=0), axis=0)
    std_train = np.expand_dims(np.nanstd(data_train_temp, axis=0), axis=0)
    del data_train_temp

    data_train_temp = np.reshape(data_all, (data_all.shape[0]*data_all.shape[1], data_all.shape[2]))
    mean_train = np.expand_dims(np.nanmean(data_train_temp, axis=0), axis=0)
    std_train = np.expand_dims(np.nanstd(data_train_temp, axis=0), axis=0)
    del data_train_temp

    #normalize training data
    for sample_i in range(num_samples_train):
        data_train[sample_i]=(data_train[sample_i]-np.repeat(mean_train, data_train.shape[1], axis=0))/\
                             np.repeat(std_train, data_train.shape[1], axis=0)
        labels_train[sample_i]=(labels_train[sample_i]-
                                np.repeat(mean_train[:,predicted_attributes], labels_train.shape[1], axis=0))/\
                               np.repeat(std_train[:,predicted_attributes], labels_train.shape[1], axis=0)
                                #normalizing also the labels which are future attributes

    #normalize validation data
    for sample_i in range(num_samples_val):
        data_val[sample_i]=(data_val[sample_i]-np.repeat(mean_train, data_val.shape[1], axis=0))/\
                             np.repeat(std_train, data_val.shape[1], axis=0)
        labels_val[sample_i]=(labels_val[sample_i]-
                              np.repeat(mean_train[:,predicted_attributes], labels_val.shape[1], axis=0))/\
                             np.repeat(std_train[:,predicted_attributes], labels_val.shape[1], axis=0)


    #normalize testing data
    for sample_i in range(num_samples_test):
        data_test[sample_i]=(data_test[sample_i]-np.repeat(mean_train, data_test.shape[1], axis=0))/\
                             np.repeat(std_train, data_test.shape[1], axis=0)
        labels_test[sample_i]=(labels_test[sample_i]-
                               np.repeat(mean_train[:,predicted_attributes], labels_test.shape[1], axis=0))/\
                             np.repeat(std_train[:,predicted_attributes], labels_test.shape[1], axis=0)
                                #normalizing also the labels which are future attributes

    #normalize entire data
    for sample_i in range(data_all.shape[0]):
        data_all[sample_i]=(data_all[sample_i]-np.repeat(mean_train, data_all.shape[1], axis=0))/\
                             np.repeat(std_train, data_all.shape[1], axis=0)
        labels_all[sample_i]=(labels_all[sample_i]-
                              np.repeat(mean_train[:,predicted_attributes], labels_all.shape[1], axis=0))/\
                             np.repeat(std_train[:,predicted_attributes], labels_all.shape[1], axis=0)
                                #normalizing also the labels which are future attributes

    with open(SaveDirParent + "preprocess_data.pkl", 'wb') as Results:  # Python 3: open(..., 'wb')
        pickle.dump(
            [data_all, labels_all, data_train, labels_train, data_val, labels_val, data_test, labels_test, mean_train,
             std_train, features_all, county_list], Results)


    ########################### training lstm model with different hyperparameters and choose the one lowest loss ######

    # Examining multiple lstm architectures
    for num_lstm_i in num_lstm_layers:
        for num_state_i in num_lstm_states:
            print('Training LSTM with ' + str(num_lstm_i) + " layers and " + str(num_state_i) + 'hidden states:')
            # Saving results parameters
            results_train = np.zeros((5, num_classes))  # saving the results (5 metrics) of 3 labels
            results_val = np.zeros((5, num_classes))  # saving the results (5 metrics) of 3 labels
            results_test = np.zeros((5, num_classes))  # saving the results (5 metrics) of 3 labels

            network = {
                "x_input_size": (None, data_train.shape[2]),
                "num_lstm_layers": num_lstm_i,
                'num_lstm_states': num_state_i,
                'num_fc_layers': 2,
                'num_fc_nodes': 512,
                "num_out_nodes": labels_train.shape[2],
                "dropout_prob": dropout_prob,
                "reg_r": lambda_loss_amount,
                'base_learning_rate': base_learning_rate,
                'decayRate': decayRate
            }
            lstm_model = model_fun(network)

            save_path_loss = SaveDirParent + "lstm_" + str(num_lstm_i) + "layers_" + str(num_state_i) +"states.h5"

            # Data generators
            training_generator = DataGenerator(data_train, labels_train, batch_size=batch_size, shuffle=True)
            validation_generator = DataGenerator(data_val, labels_val, batch_size=data_val.shape[0], shuffle=False)
            # Callback to save the model with best loss
            checkpoint_best = ModelCheckpoint(
                save_path_loss, monitor='val_loss', verbose=0, save_best_only=True, mode='min')#, save_freq='epoch'
            history = lstm_model.fit(x=training_generator, validation_data=validation_generator, epochs=num_epochs,
                                     callbacks=checkpoint_best)

            # Loading the best model based on the validation data
            lstm_model = load_model(save_path_loss)

            #training results
            labels_train_pred = lstm_model.predict(data_train)
            for label_i in range(labels_train.shape[2]):
                results_train_temp = np.zeros((5, labels_train.shape[0]))
                for sample_i in range(labels_train.shape[0]):
                    results_train_temp[:, sample_i] = calculate_results(labels_train[sample_i, :, label_i],
                                                                    labels_train_pred[sample_i, :, label_i],
                                                                    mean_train[:,predicted_attributes[label_i]],
                                                                    std_train[:,predicted_attributes[label_i]])
                results_train[:, label_i] = np.nanmean(results_train_temp, axis=1)
                del results_train_temp

            for label_i in range(labels_train.shape[2]):
                print(
                    "Final training results of label %d: RMSE %.2f, MAE %.2f, "
                    "Correlation coefficient %.2f (p=%.4f)." % (
                    label_i, results_train[0,label_i],
                    results_train[1, label_i],
                    results_train[3, label_i],
                    results_train[4, label_i]))


            #validation results
            labels_val_pred = lstm_model.predict(data_val)
            for label_i in range(labels_val.shape[2]):
                results_val_temp = np.zeros((5, labels_val.shape[0]))
                for sample_i in range(labels_val.shape[0]):
                    results_val_temp[:, sample_i] = calculate_results(labels_val[sample_i, :, label_i],
                                                                    labels_val_pred[sample_i, :, label_i],
                                                                    mean_train[:,predicted_attributes[label_i]],
                                                                    std_train[:,predicted_attributes[label_i]])
                results_val[:, label_i] = np.nanmean(results_val_temp, axis=1)
                del results_val_temp

            for label_i in range(labels_val.shape[2]):
                print(
                    "Final validation results of label %d: RMSE %.2f, MAE %.2f, "
                    "Correlation coefficient %.2f (p=%.4f)." % (
                    label_i, results_val[0,label_i],
                    results_val[1, label_i],
                    results_val[3, label_i],
                    results_val[4, label_i]))


            #test results
            labels_test_pred = np.zeros(labels_test.shape)
            for day_i in range(adapt_train, data_test.shape[1]+adapt_train, adapt_train):
                    #+adapt_train to consider last days of testing
                labels_all_pred = lstm_model.predict(data_all[:, :test_date_ind+day_i])
                labels_test_pred [:, day_i-adapt_train:day_i, :]= labels_all_pred[:,
                                                                  test_date_ind+day_i-adapt_train:test_date_ind+day_i, :]

                #adapt the model on the data until n weeks ago of testing day where the labels already known from
                    # current testing day
                # Data generators
                training_generator = DataGenerator(data_all[train_county_inds, :test_date_ind+day_i-prediction_day],
                                                   labels_all[train_county_inds, :test_date_ind+day_i-prediction_day],
                                                   batch_size=batch_size, shuffle=True)
                validation_generator = DataGenerator(data_all[val_county_inds, :test_date_ind+day_i-prediction_day],
                                                     labels_all[val_county_inds, :test_date_ind+day_i-prediction_day],
                                                     batch_size=data_val.shape[0], shuffle=False)
                # Callback to save the model with best loss
                checkpoint_best = ModelCheckpoint(
                    save_path_loss, monitor='val_loss', verbose=0, save_best_only=True, mode='min')  # , save_freq='epoch'
                history = lstm_model.fit(x=training_generator, validation_data=validation_generator, epochs=num_epochs_adapt,
                                         callbacks=checkpoint_best, verbose=0)
                lstm_model = load_model(save_path_loss)

            for label_i in range(labels_test.shape[2]):
                results_test_temp = np.zeros((5, labels_test.shape[0]))
                for sample_i in range(labels_test.shape[0]):
                    results_test_temp[:, sample_i] = calculate_results(labels_test[sample_i, :, label_i],
                                                                    labels_test_pred[sample_i, :, label_i],
                                                                    mean_train[:,predicted_attributes[label_i]],
                                                                    std_train[:,predicted_attributes[label_i]])
                results_test[:, label_i] = np.nanmean(results_test_temp, axis=1)
                del results_test_temp


            for label_i in range(labels_test.shape[2]):
                print(
                    "Final testing results of label %d: RMSE %.2f, MAE %.2f, "
                    "Correlation coefficient %.2f (p=%.4f)." % (
                    label_i, results_test[0,label_i],
                    results_test[1, label_i],
                    results_test[3, label_i],
                    results_test[4, label_i]))

            ###############Saving the variables#################
            # Saving the objects:
            with open(SaveDirParent + "lstm_" + str(num_lstm_i) + "layers_" + str(num_state_i) +
                      "states_results.pkl", 'wb') as Results:  # Python 3: open(..., 'wb')
                pickle.dump(
                    [results_train, results_val, results_test, labels_train_pred, labels_val_pred, labels_test_pred,
                     history.history, labels_all_pred], Results)

            clear_session()

    ########################### Finding the best LSTM model based on the validation data ######

    with open(SaveDirParent+'preprocess_data.pkl', 'rb') as data_combined:
        [data_all, labels_all, data_train, labels_train, data_val, labels_val, data_test, labels_test,
         mean_train, std_train, features_all, county_list] = pickle.load(data_combined)

    #Finding minimum validation MAE of label 0 (total cases)
    min_val_mae = 1000 #initial value
    label_i = 0
    for num_lstm_i in num_lstm_layers:
        for num_state_i in num_lstm_states:
            with open(SaveDirParent + "lstm_" + str(num_lstm_i) + "layers_" + str(num_state_i) +"states_results.pkl", 'rb') as results:
                [results_train, results_val, results_test, labels_train_pred, labels_val_pred, labels_test_pred,
                     history, labels_all_pred] = pickle.load(results)
            if results_val[1, label_i] < min_val_mae:
                num_lstm_best = num_lstm_i
                num_state_best = num_state_i
                min_val_mae = results_val[1, label_i]
    print('\n\n')
    print('Best validation is for LSTM with ' + str(num_lstm_best) + " layers and " + str(num_state_best) + ' hidden states:')
    with open(SaveDirParent + "lstm_" + str(num_lstm_best) + "layers_" + str(num_state_best) + "states_results.pkl",
              'rb') as results:
        [results_train, results_val, results_test, labels_train_pred, labels_val_pred, labels_test_pred,
         history, labels_all_pred] = pickle.load(results)

    for label_i in range(labels_train.shape[2]):
        print(
            "Final training results of label %d: RMSE %.2f, MAE %.2f, Correlation coefficient %.2f (p=%.4f)." % (
                label_i, results_train[0, label_i],
                results_train[1, label_i],
                results_train[3, label_i], results_train[4, label_i]))

    for label_i in range(labels_val.shape[2]):
        print(
            "Final validation results of label %d: RMSE %.2f, MAE %.2f, Correlation coefficient %.2f (p=%.4f)." % (
                label_i, results_val[0, label_i],
                results_val[1, label_i],
                results_val[3, label_i], results_val[4, label_i]))

    for label_i in range(labels_test.shape[2]):
        print(
            "Final testing results of label %d: RMSE %.2f, MAE %.2f, Correlation coefficient %.2f (p=%.4f)." % (
                label_i, results_test[0, label_i],
                results_test[1, label_i],
                results_test[3, label_i], results_test[4, label_i]))

