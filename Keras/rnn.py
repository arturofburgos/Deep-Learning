# Algorítmo: Recursive Neural Network
# Professor: Aldemir Aparecido Cavalini Júnior
# Aluno: Arturo Burgos


# OBS --> Comentários: da seguinte forma #"exemplo" foram testes que fiz durante o código, caso queira vê-los basta apagá-los

#----------------------------------------------------------------------------------------------------------------------------------#
# Importar as bibliotecas do numpy, keras e scipy

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from scipy import signal
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------------------------#
# Ler arquivo .csv

data = np.loadtxt("1e-1s.csv") 


i_resp = 7 # Ceta início das respostas, por estarem posicionados na coluna 7 -->Definição de qual resposta trabalharei
i_resp = i_resp + 6 # Escolha de uma das respostas - 1 a 7

#----------------------------------------------------------------------------------------------------------------------------------#
# Filtragem

media = np.mean(data[:,i_resp]) # Calcula a média da coluna com indice i_resp
data[:,i_resp] = data[:,i_resp] - media # Retira media calculada da resposta original



#plt.plot(data[0:1000,0],data[0:1000,i_resp]) BBBB
#plt.show()



b, a = signal.butter(2,.06,btype="lowpass") # Parâmetros de filtrgem de sinal lowpass - passa baixa
# Aparentemente b e a são os coeficientes dos vetores filtro (IIR filter)
# scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None) --> no caso é filtro digital
# N - a ordem do filtro, ex. 2, 3, 4...
# Wn - Para filtros analógicos a frequência ou as frequências críticas de filtro, ex. 100 hz num passa baixa faz com que valores de frequência maiores
# (Contiuação linha anterior) sejam filtrados, para filtros digitais é normalizado entre 0 e 1 onde 1 é a frequência de 
# (Contiuação linha anterior) Nyquist
# Vale lembrar que essa chamada só define o filtro, instancia, para a filtragem propriamente dita temos o comando signal.lfilter



#print(data[0:3,i_resp]) # Conferênca do vetor AAAA



data[:,i_resp] = signal.lfilter(b,a,data[:,i_resp]) # Filtragem propriamente dita, de acordo com a documentação pode filtrar em 
# (Continuação linha anterior) IIR ou FIR - Impulse Response Infinite and Finite - Ver as implicações com o Professor Aldemir
# (Idem) em relação à do por que o IIR está sendo usado. IIR normalmente rápido, menor estabilidade e de baixa ordem. FIR alta ordem,
# (Idem) lento e muito estável.
# scipy.signal.lfilter(b, a, x, axis=- 1, zi=None)
# b - o vetor de coeficiente do "numerador" obtidos com o signal.butter
# a - o vetor de coeficiente do "denominador" obtidos com o signal.butter
# axis - se é linha ou coluna por default vai coluna como -1 - Ao menos acho isso, verificar com Professor Aldemir e Denise
# zi - Condições iniciais do filtro, é um vetor, no caso o próprio data[:,i_resp]



#print(data[0:3,i_resp]) # Conferência do vetor já filtrado AAAA



#plt.plot(data[0:1000,0],data[0:1000,i_resp]) BBBB
#plt.show()



#----------------------------------------------------------------------------------------------------------------------------------#
# Criação das matrizes x_input e y_out. Por fim, atribuição de parte da data (.csv)

m,n = data.shape # Atribuindo às variáveis m e n o tamanho do arquivo
# m - número de linhas -> 18000
# n - número de colunas -> 16


i_gap = 2 # Quantos atrasos são treinados - Palavras do Prof. Duarte ---> tentar ver se dá para não usar 
n_input = 6 + i_gap
n_results = m - i_gap
x_input=np.zeros([n_results,n_input]) # Crio uma matriz com n_results linhas e n_input colunas



#----------------------------------------------------------------------------------------------------------------------------------#
# Teste a fim de verificar se entendi corretamente o for abaixo 

#x_input[0,0:6]=data[0+i_gap,1:7]
#print(x_input[0,0:8])
#x_input[0,6+0]=data[0+0,i_resp]
#print(x_input[0,0:8])

#x_input[0,0:6]=data[0+i_gap,1:7]
#print(x_input[0,0:8])
#x_input[0,6+1]=data[0+1,i_resp]
#print(x_input[0,0:8])

#x_input[1,0:6]=data[1+i_gap,1:7]
#print(x_input[1,0:8])
#x_input[1,6+0]=data[1+0,i_resp]
#print(x_input[1,0:8])

#x_input[1,0:6]=data[1+i_gap,1:7]
#print(x_input[1,0:8])
#x_input[1,6+1]=data[1+1,i_resp]
#print(x_input[1,0:8])

#print("\n\n")
#print(x_input[0,0:8])
#print(x_input[1,0:8])
#----------------------------------------------------------------------------------------------------------------------------------#

for i in range(n_results):
    x_input[i,0:6]= data[i + i_gap,1:7] # Aqui copia-se os dados de 1 a 6 da variavel data (.csv) e coloca-os no x_input
    # (Continuação) importante notar que com a presença do gap a coleta de dados acontece duas linhas adiante do início
    for j in range(i_gap):
        x_input[i,6 + j]= data[i + j,i_resp] # Aqui copia-se os dados da coluna i_resp já filtrados (no programa "RecursiveNeuralNetwork"
        # (Continuação) não filtrei os dados, ver com Professor Aldemir e Denise quanto a isso) agora aqui o i_gap é utilizado para
        # (Continuação) salvar a informação atual do i_resp na coluna 7. Por sua vez na coluna 8 salva o i_resp do proximo dt e 
        # (Continuação) assim por diante. A pergunta é:por que salvar dois i_resp de tempos distintos numa mesma célula da matriz?

#print(x_input[0:3,0:8])


y_out = data[i_gap:m,i_resp] # Criação do vetor y_out atribuindo o valores de data (.csv)

#lx,cx = x_input.shape
#print(lx)
#print(cx)
#print(len(y_out))

#----------------------------------------------------------------------------------------------------------------------------------#
# Treino e teste

x_train=np.array(x_input[0:5000,:]) # Definindo 5000 instantes de tempo para treinamento comtemplando todas as colunas. Pregunta:
# (Continuação) por que contemplar todas as colunas, isso recai na duvida do segundo for? Ver com Professor Aldemir e Denise 
y_train=np.array(y_out[0:5000]) # Definindo 5000 instantes de tempo de treinamento
x_test=np.array(x_input[5000:n_results,:]) # Definindo o restante do  conjunto para teste
y_test=np.array(y_out[5000:n_results]) #  Definindo o restante do  conjunto para teste

#----------------------------------------------------------------------------------------------------------------------------------#
# Definindo o modelo do Keras ---> Dúvidas aqui 

model = Sequential() # Definindo a classe do modelo, no caso sequencial
model.add(Dense(n_input, input_dim=n_input, activation='linear'))
model.add(Dense(12, activation='linear'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(1, activation='linear'))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(x_train, y_train,batch_size = 64,epochs=10000,use_multiprocessing=True )
y_predict = model.predict(x_test)
np.savetxt("y_predict.txt",y_predict+media)
np.savetxt("y_real.txt",y_test+media)