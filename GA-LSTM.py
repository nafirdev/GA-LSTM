from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
import random
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def create_lstm_model(input_shape,lstm_units,dropout_rate):
    model = Sequential()
    model.add(LSTM(lstm_units,input_shape= input_shape,return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(5))
    optimizer= Adam(learning_rate=0.001,clipnorm=1.0)
    model.compile(optimizer=optimizer,loss='mean_squared_error')
    return model


def initialize_population(population_size , parameter_space):
    population=[]
    for _ in range(population_size):
        invidual = [random.choice(parameter_space[key]) for key in parameter_space.keys()]
        population.append(invidual)
        return population

def crossover(parent1,parent2):
    crossover_point = random.randint(1,len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(invidual,mutation_rate,parameter_space):
    for i in range(len(invidual)):
        if random.random() < mutation_rate:
          parameter_name = list(parameter_space.keys())[i]
          invidual[i]= random.choice(parameter_space[parameter_name])
    return invidual

def evaluate_inviduals(invidual1,x_train,y_train,x_val,y_val):
    lstm_units,dropout_rate,batch_size,epochs= invidual1
    model =create_lstm_model(x_train.shape[1:], lstm_units ,dropout_rate)
    early_stopping=EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
    model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_val,y_val),callbacks=[early_stopping],verbose=0)
    y_pred=model.predict(x_val)
    mse=mean_squared_error(y_val,y_pred)
    return mse

def optimize_lstm_parameters(x_train,y_train,x_val,y_val,parameter_space,population_size,num_generations):
    population = initialize_population(population_size,parameter_space)
    for generatin in range(num_generations):
        fitness_scores=[]
        for invidual in population:
            fitness = evaluate_inviduals(invidual,x_train,y_train,x_val,y_val)
            fitness_scores.append((invidual,fitness))

        fitness_scores.sort(key=lambda x: x[1])

        selected_parents= [invidual for invidual, _ in fitness_scores[:population_size // 2]]

        next_generation = selected_parents.copy()
        while len(next_generation) < population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)
            child=crossover(parent1,parent2)
            child = mutate(child , mutation_rate=0.1 , parameter_space=parameter_space)
            next_generation.append(child)

        population = next_generation

    best_invidual , _ = min(fitness_scores, key =lambda x: x[1])
    return best_invidual


parameter_space = {'lstm_units': [16,32,64,128] , 'dropout_rate': [0.2,0.3,0.4,0.5] , 'batch_size': [16,32,64,128] , 'epochs': [10,20,30,40,50]}


best_parameters= optimize_lstm_parameters(xtrain,ytrain,x_test,y_test,parameter_space,population_size=10,num_generations=7)
print('Best Parameters is :', best_parameters)

