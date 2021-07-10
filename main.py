print('=' * 10, '[ 🚇 지하철 혼잡도 예측 프로그램 🚇 ]', '=' * 10, '\nLoading...')

import numpy  as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from lib import loadDataPandas

tf.disable_v2_behavior()
plt.rc('font', family='Malgun Gothic')

print(f'{"=" * 10} [ 모듈 로딩 완료 ✔ ] {"=" * 10}')

column_names = [
  '사용월',
  '호선명',
  '지하철역',
  '00시-01시 승차인원',
  '00시-01시 하차인원',
  '01시-02시 승차인원',
  '01시-02시 하차인원',
  '02시-03시 승차인원',
  '02시-03시 하차인원',
  '03시-04시 승차인원',
  '03시-04시 하차인원',
  '04시-05시 승차인원',
  '04시-05시 하차인원',
  '05시-06시 승차인원',
  '05시-06시 하차인원',
  '06시-07시 승차인원',
  '06시-07시 하차인원',
  '07시-08시 승차인원',
  '07시-08시 하차인원',
  '08시-09시 승차인원',
  '08시-09시 하차인원',
  '09시-10시 승차인원',
  '09시-10시 하차인원',
  '10시-11시 승차인원',
  '10시-11시 하차인원',
  '11시-12시 승차인원',
  '11시-12시 하차인원',
  '12시-13시 승차인원',
  '12시-13시 하차인원',
  '13시-14시 승차인원',
  '13시-14시 하차인원',
  '14시-15시 승차인원',
  '14시-15시 하차인원',
  '15시-16시 승차인원',
  '15시-16시 하차인원',
  '16시-17시 승차인원',
  '16시-17시 하차인원',
  '17시-18시 승차인원',
  '17시-18시 하차인원',
  '18시-19시 승차인원',
  '18시-19시 하차인원',
  '19시-20시 승차인원',
  '19시-20시 하차인원',
  '20시-21시 승차인원',
  '20시-21시 하차인원',
  '21시-22시 승차인원',
  '21시-22시 하차인원',
  '22시-23시 승차인원',
  '22시-23시 하차인원',
  '23시-24시 승차인원',
  '23시-24시 하차인원',
]

dataset = loadDataPandas.read('./dataset/subway.csv', column_names)
stations = list(set(dataset['지하철역'].to_numpy()))


class App:
  def __init__(self, station):
    self.__station = station
    self.select_dataset = dataset[dataset['지하철역'] == station][column_names[3:]].to_numpy()

  def findRideAndQuitData(self):
    if stations.count(self.__station) == 0:
      return 0
    new_data = [[], []]  
    for data in self.select_dataset:
      ride = []
      quit = []
      for i in range(48):
        if i % 2 == 0:
          quit.append(data[i])
          continue
        ride.append(data[i])
      new_data[0].append(ride)
      new_data[1].append(quit)
    return new_data

  def drawRideAndQuitGraph(self):
    _data = self.findRideAndQuitData()
    if _data != 0:
      for i in range(len(_data[0])):
        plt.plot(_data[0][i], _data[1][i], 'ro')
      plt.title(f'{self.__station}의 승·하차 인원 수 관계')
      plt.xlabel('승차 인원')
      plt.ylabel('하차 인원')
      plt.show()
    else:
      print("해당 역의 데이터를 조회할 수 없습니다.")

  def trainingRideAndQuitModel(self):
    _data = self.findRideAndQuitData()
    x_data = np.ravel(_data[0], order='C')
    y_data = np.ravel(_data[1], order='C')
    X = tf.placeholder(tf.float32, shape=None)
    Y = tf.placeholder(tf.float32, shape=None)
    W = tf.Variable(tf.random_uniform([1], -100, 100), 'weight')
    b = tf.Variable(tf.random_uniform([1], -100, 100), 'bias')
    H = X * W + b
    cost = tf.reduce_mean(tf.square(H - Y))
    optimizer = tf.train.GradientDescentOptimizer(tf.Variable(0.00000000001))
    train = optimizer.minimize(cost)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    for step in range(100001):
      session.run(train, feed_dict = { X: x_data, Y: y_data })
      if step % 10000 == 0:
        print(step, session.run(cost, feed_dict = { X: x_data, Y: y_data }), session.run(W), session.run(b))
    return session

  def findCongestionData(self):
    new_data = []
    for data in self.select_dataset:
      layer = []
      for i in range(0, 48, 2):
        ride = data[i]
        quit = data[i + 1]
        layer.append(ride + quit)
      new_data.append(layer)
    return new_data

  def drawCongestionGraph(self):
    _data = self.findCongestionData()
    for arr in _data:    plt.plot(arr, 'ro')
    plt.title(f'{self.__station}의 시간대별 혼잡도')
    plt.xlabel('시간대')
    plt.ylabel('혼잡도')
    plt.show()


while 1:
  station = input('\n👉 지금 어디 역에 계시나요? (종료: 0)\n')
  if station == '0':  break
  app = App(station)
  app.drawCongestionGraph()
