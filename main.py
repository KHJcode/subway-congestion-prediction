getShellText = lambda text: "=" * 10 + text + "=" * 10
print(getShellText('[ 🚇 지하철 혼잡도 예측 프로그램 🚇 ]'), '\nLoading...')

import numpy  as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from datetime import datetime
from utils import loadDataPandas

tf.disable_v2_behavior()
plt.rc('font', family='Malgun Gothic')

print(getShellText('[ 모듈 로딩 완료 ✔ ]'))

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
slot_names = column_names[3:]

dataset = loadDataPandas.read('./dataset/subway.csv', column_names)
stations = list(set(dataset['지하철역'].to_numpy()))


class App:
  def __init__(self, station):
    self.__station = station
    self.select_dataset = dataset[dataset['지하철역'] == station][slot_names].to_numpy()
    self.X = tf.placeholder(tf.float32, shape=None)
    self.Y = tf.placeholder(tf.float32, shape=None)
    self.W = tf.Variable(tf.random_uniform([1], -100, 100), 'weight')
    self.b = tf.Variable(tf.random_uniform([1], -100, 100), 'bias')
    self.H = self.X * self.W + self.b
    self.cost = tf.reduce_mean(tf.square(self.H - self.Y))
    optimizer = tf.train.GradientDescentOptimizer(tf.Variable(0.001))
    self.train = optimizer.minimize(self.cost)
    self.congestion_session = 0
    self.x_data = list(range(24))
    self.y_data = []
    self.slots  = []
    self.getString = lambda x: '0' + str(x) if x < 10 else str(x)
    self.now = 0

  def findRideAndQuitData(self):
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
    for i in range(len(_data[0])):
      plt.plot(_data[0][i], _data[1][i], 'ro')
      plt.title(f'{self.__station}의 승·하차 인원 수 관계')
      plt.xlabel('승차 인원')
      plt.ylabel('하차 인원')
      plt.show()

  '''
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
  '''

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
  
  def trainingCongestionModel(self):
    select_dataset = dataset[dataset['지하철역'] == self.__station][slot_names]
    new_data = {}
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    for i in range(0, 48, 2):
      new_data[slot_names[i][:7]] = sum(
        select_dataset[
          [
            slot_names[i],
            slot_names[i + 1]
          ]
        ].sum().to_numpy()
      ) // len(select_dataset)
    new_data = sorted(new_data.items(), key=lambda x:x[1])
    for i in range(len(new_data)):
      self.y_data.append(new_data[i][1])
      self.slots.append(new_data[i][0])
    for step in range(10000):
      session.run(
        self.train,
        feed_dict = {
          self.X: self.x_data,
          self.Y: self.y_data
        },
      )
      if step % 1000 == 0:  print('...')
    self.congestion_session = session

  def getNowSlotIndex(self):
    self.now = datetime.now()
    hour = self.now.hour
    return self.slots.index(
      f'{self.getString(hour)}시-{self.getString(hour + 1)}시'
    )

  def predictionCongestion(self):
    if self.congestion_session == 0:  return 0
    i = self.getNowSlotIndex()
    result = self.congestion_session.run(self.H, feed_dict={ self.X: [i] })[0]
    average = sum(self.y_data) // 24
    print('_' * 10)
    print(f'\n[⏰ {self.getString(self.now.hour)}:{self.getString(self.now.minute)}]\n')
    print(f'● 예측 혼잡도 : {result}')
    print(f'● {"혼잡한 시간대입니다." if average < result else "여유로운 시간대입니다."}')
    print('_' * 10)
    


while 1:
  station = input('\n👉 지금 어디 역에 계시나요? (종료: 0)\n')
  if station == '0':  break
  if stations.count(station) != 0:
    app = App(station)
    print('Loading...')
    app.trainingCongestionModel()
    print(getShellText('[ 학습 완료 ]'))
    app.predictionCongestion()
    app.drawCongestionGraph()
    continue
  print('=> 해당 역의 데이터를 조회할 수 없습니다.')
