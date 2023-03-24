#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pkl
import random
import datetime as dt
import seaborn as sns


# In[ ]:


class goal_program():
    
    def __init__(self, name='goals'):
        self.name = name
        self.goals = {}
        self.gl_hist = {}
        
    def add_goal(self, goal_name='meditate', stdt = dt.datetime.today()):
        self.goals[goal_name] = betabinomial(goal_name, stdt)
        self.goals[goal_name].setup()
        self.goals[goal_name].find_prior_values()
        self.goals[goal_name].update_prior()
        self.goals[goal_name].update_data()
        self.goals[goal_name].plot_model()
        
    def update_data(self, kind='soft'):
        if kind=='soft':
            for gl in self.goals:
                if ((dt.datetime.today()-self.goals[gl].last_update).days > self.goals[gl].days_til_update) &                 (((dt.datetime.today()-self.goals[gl].last_update).days) >= 1):
                    print(self.goals[gl].name)
                    self.goals[gl].add_new_rows()
                    self.goals[gl].update_data()
                    self.goals[gl].update_pos()
                    self.goals[gl].dump_data()
        else:
            for gl in self.goals:
                print(self.goals[gl].name)
                self.goals[gl].add_new_rows()
                self.goals[gl].update_data()
                self.goals[gl].update_pos()
                self.goals[gl].dump_data()
            
    def srt_plot_dot(self):
        goal_dict = {}
        for gl in self.goals:
            self.goals[gl].update_pos()
            goal_dict[gl] = self.goals[gl].return_prob_of_goal()
        today = str(dt.datetime.today())[:10]
        self.gl_hist[today] = goal_dict
        sdd = sorted(goal_dict.items(), key=lambda kv: kv[1])
        numgoals = len(sdd)
        rows = round(np.sqrt(numgoals))
        cols = rows+1
        plt.figure(figsize=(15,8))
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.axis('off')
        plt.title('My Goals', size='xx-large', loc='left')
        exes = list(np.linspace(.1,.9,len(sdd)))
        for i, goal in enumerate(sdd):
            order = i*2
            color = goal[1]
            if color < .2:
                red = np.linspace(.75,1,10)[int(color/.0201)]
                green = np.linspace(0,.5,10)[int(color/.0201)]
                blue = 0
                size = .95 - color*.25
            elif (color >= .2) & (color <.4):
                red = np.linspace(1,.95,10)[int((color-.2)/.0201)]
                green = np.linspace(.5,.95,10)[int((color-.2)/.0201)]
                blue = np.random.rand()*.4
                size = .95 - color*.25
            elif (color >= .4) & (color < .8):
                red = np.linspace(.95,.4,10)[int((color-.4)/.0601)]
                green = np.linspace(.95,.8,10)[int((color-.4)/.0601)]
                blue = np.random.rand()*.5
                size = .95 - color*.25
            else:
                red = np.linspace(.95,.4,10)[int((color-.4)/.0601)]
                green = np.linspace(.95,.8,10)[int((color-.4)/.0601)]
                blue = np.random.rand()*.5
                size = .75 - ((color-.8)*2)
            myx = np.random.choice(exes)
            exes.remove(myx)
            x = myx + np.random.choice([-.02,-.01,0,.01,.02])
            y = .15+color*.75+np.random.rand()*.05
            ## plot balloon buckets
            plt.plot(x-(size*.007),y-(size*.09), '|', markersize=20*size, c='black')
            plt.plot(x+(size*.007),y-(size*.09), '|', markersize=20*size, c='black')
            plt.plot(x,(y-(.125*size)), 's', markersize=16*size, c=(.6,.3,.2))
            plt.plot(x,y, 'o', markersize=(80*size), c=(red,green,blue))
            plt.text(x-.03, y-.003, goal[0])
#             self.goals[goal].days_til_update = color*3
        ## plot some random trees

        for i in range(54):
            tx = np.random.rand()
            dy = np.random.rand()
            plt.plot(tx,.09+dy*.02,'s', markersize=8, c=(.6,.3,.2))
            plt.plot(tx,.13+dy*.02, '^', markersize=30, c=(.15+np.random.rand()*.3,.4+np.random.rand()*.3,0))
        for i in range(18):
            tx = np.random.rand()
            dy = np.random.rand()
            plt.plot(tx,.05+dy*.015,'s', markersize=10, c=(.6,.3,.2))
            plt.plot(tx,.10+dy*.015, '^', markersize=40, c=(.15+np.random.rand()*.3,.4+np.random.rand()*.3,0))
        for i in range(6):
            tx = np.random.rand()
            dy = np.random.rand()
            plt.plot(tx,.02+dy*.015,'s', markersize=13, c=(.6,.3,.2))
            plt.plot(tx,.09+dy*.015, '^', markersize=55, c=(.15+np.random.rand()*.3,.4+np.random.rand()*.3,0))
        for i in range(2):
            tx = np.random.rand()
            dy = np.random.rand()
            plt.plot(tx,.02,'s', markersize=17, c=(.6,.3,.2))
            plt.plot(tx,.12, '^', markersize=70, c=(.15+np.random.rand()*.3,.4+np.random.rand()*.3,0))
            
    def store_goal(self, gl):
        dumpstr = gl + '_goal.pkl'
        with open(dumpstr, 'wb') as file:
            pkl.dump(self.goals[gl], file)
        file.close()
        
    def read_data(self, gl):
        readstr = gl + '_goal.pkl'
        with open(readstr, "rb") as file:
            self.goals[gl] = pkl.load(file)
        file.close()
    
    def create_goal_from_data(self, gl):
        readstr = gl + '_data.pkl'
        with open(readstr, "rb") as file:
            [data, stdt, k, n, days_history,
                      g, gd, goal_mult, name, d, goal_type] = pkl.load(file)
        file.close()
        if goal_type == 'betabinomial':
            self.goals[gl] = betabinomial(name=name, start_date=stdt)
            self.goals[gl].data = data
            self.goals[gl].last_update = max(data['date'])
            self.goals[gl].k = k
            self.goals[gl].n = n
            self.goals[gl].days_history = days_history
            self.goals[gl].g = g
            self.goals[gl].gd = gd
            self.goals[gl].goal_mult = goal_mult
            self.goals[gl].d = d
            self.goals[gl].days_til_update = 1
            self.goals[gl].update_prior()
            self.goals[gl].update_pos()
            self.goals[gl].update_data()


# In[ ]:


def create_df(stdt, periods=1):
        return pd.DataFrame({'date':pd.date_range(stdt, periods=periods), 'n':1,'k':np.nan})


# In[ ]:


class betabinomial():
   
    def __init__(self, name='name', 
                 start_date = str(dt.datetime.today())[:10],
                 freq='day'):
        self.goal_type = 'betabinomial'
        self.name = name
        self.stdt = pd.to_datetime(start_date) - dt.timedelta(days=1)
        self.n = 0
        self.k = 0  ## feed smarter priors
        self.a = self.k+1
        self.b = self.n-self.k+1
        self.last_update = pd.to_datetime(start_date) - dt.timedelta(days=1)
        self.days_history = 180
        self.data = create_df(start_date)
        
#     def create_df(self, periods=1):
#         return pd.DataFrame({'date':pd.date_range(self.stdt, periods=periods), 'n':1,'k':np.nan})
    
    def find_prior_values(self):
        
        print('In 30 days prior to {}, how many times did you {}?'.format(str(self.stdt)[:10], self.d))
        p = float(input())
        p = p/30
        print('On a scale of 1 to 10, how sure are you about this estimate?')
        strength = float(input())
        if strength <2:
            print('strength = 1, so setting values to uninformative prior.')
            n=0
            k=0
        else:
            s = int(strength)+int(abs(.5-p)*30)+1
            n=s
            k=round(s*p)
            self.n = n
            self.k = k
        
    def update_prior(self):
        self.a = self.k+1
        self.b = self.n-self.k+1
        self.modpr = st.beta(self.a, self.b)
        self.xpri = np.linspace(self.modpr.ppf(0.01), self.modpr.ppf(0.99), 50)
        
    def add_new_rows(self):
        today=dt.datetime.today()
        days_since = (today - self.last_update).days - 1
        strtdt = (self.last_update + dt.timedelta(days=1))
        newdf = create_df(stdt=strtdt, periods=days_since)
        self.data = pd.concat([self.data,newdf], axis=0)
        self.data.reset_index(inplace=True)
        self.data.drop('index', axis=1, inplace=True)
        self.last_update = max(self.data['date'])
        
    def update_data(self):
        miss_cnt = np.isnan(self.data['k']).sum()
        if miss_cnt >= 1:
            print(self.data.iloc[-7:])
            print('It has been {} day(s). How many times did you {} since {}?.'.format(miss_cnt, self.d, str(self.last_update)[:10]))
            newk1 = int(input())
            if newk1 > miss_cnt:
                newk1 = miss_cnt
            newk2 = np.concatenate((np.zeros(miss_cnt-newk1), np.ones(newk1)), axis=None)
            self.data.loc[self.data.index.stop-miss_cnt:,'k'] = newk2
            self.update_pos()
            self.days_til_update = st.poisson.rvs((self.return_prob_of_goal()*4), size=1)[0]
        
    def setup(self):
#         self.data = create_df(self.stdt)
        self.add_new_rows()
        print('What is the thing you are tracking with this goal? Plain verb first -  E.g. "cook a great meal", "exercise", "meditate".')
        self.d = input()
        print('When you {}, do you think of it as a "good" thing or a "bad" thing?'.format(self.d))
        self.good_thing = np.sum('good'==input('Enter "good" or "bad".'))
        print('Are you tracking this per "week", or "month", "quarter", or "year"?')
        self.freq1 = input()
        if self.freq1 not in ['week', 'month', 'quarter', 'year']:
              print('Please enter "week", "month", "quarter", or "year"?')
              self.freq1 = input()
        freqmap = {'week': 7, 'month': 30, 'quarter': 91, 'year':365}
        self.freq = freqmap[self.freq1]
        if self.good_thing == 1:
            self.gd = 'more'
            self.goal_mult = 1
            print("How many times per {} do you hope to {}?".format(self.freq1, self.d))
        else:
            self.gd = 'less'
            self.goal_mult = -1
            print('You want to limit the number of times you {} per {}.'.format(self.d, self.freq1))
            print('How many times per {} is a good goal?'.format(self.freq1))
        self.g = (int(input())/self.freq)
        self.days_til_update = 1
        if (self.g < .6) & (self.g > .4):
            self.days_history = 90
        else:
            self.days_history = int(900*abs(.5-self.g))
            
    def update_pos(self):
        if self.days_history > self.data.shape[0]:
            self.apos = self.k+1+np.sum(self.data['k'])
            self.bpos = self.n-self.k+np.sum(self.data['n'])-np.sum(self.data['k'])+1
        else:
            self.apos = np.sum(self.data[-self.days_history:]['k']+1)
            self.bpos = np.sum(self.data[-self.days_history:]['n']-self.data[-self.days_history:]['k']+1)
        self.modpos = st.beta(self.apos, self.bpos)
        self.xpos = np.linspace(self.modpos.ppf(0.01), self.modpos.ppf(0.99), 50)
        
    def plot_model(self):
        self.update_pos()
        plt.plot(self.xpri, self.modpr.pdf(self.xpri), '-', lw=2, label='prior');
        plt.plot(self.xpos, self.modpos.pdf(self.xpos), '-', lw=2, label='posterior')
        plt.plot([self.g,self.g],[0, max(self.modpos.pdf(self.xpos))], '--', lw=2, color='red', label='goal')
        plt.fill_between(
            x= self.xpos, 
            y1= self.modpos.pdf(self.xpos), 
            where = (self.g*self.goal_mult < self.xpos*self.goal_mult),
            color= 'grey',
            alpha= 0.2)
        plt.title(self.name)
        plt.legend();
            
    def read_data(self):
        readstr = self.name + '_data.pkl'
        with open(readstr, "rb") as file:
            self.data = pkl.load(file)
        file.close()
                                        
    def plot_posterior_model(self):
        xpos=np.linspace(self.posterior_model.ppf(0.01), self.posterior_model.ppf(0.99), 50)
        plt.plot(xpos, self.posterior_model.pdf(xpos), '-',                 lw=2, label = self.name+' + posterior', color=teamcolors[self.name]);
    
    def return_prob_of_goal(self):
        if self.gd == 'more':
            return 1-self.modpos.cdf(self.g)
        else:
            return self.modpos.cdf(self.g)
        
    def dump_data(self):
        dumpstr = self.name + '_data.pkl'
        things = [self.data, self.stdt, self.k, self.n, self.days_history,
                  self.g, self.gd, self.goal_mult, self.name, self.d, self.goal_type]
        with open(dumpstr, 'wb') as file:
            pkl.dump(things, file)
        file.close()


# In[ ]:


# mygoals = goal_program()


# In[ ]:


# list_of_goals = ['meditate','stretch','appa teeth','strength','planning','sexytimes','vices','Alpha','Boone','dates','friend','exercise','call parents', 'fasting']


# In[ ]:


# for gl in list_of_goals:
#     mygoals.create_goal_from_data(gl)


# In[ ]:


readstr = 'mygoals_program' + '_data.pkl'
with open(readstr, 'rb') as file:
    mygoals = pkl.load(file)
file.close()


# In[ ]:


mygoals.add_goal('yard', stdt='2023-03-23')


# In[ ]:


mygoals.update_data(kind='soft')


# In[ ]:


mygoals.srt_plot_dot()


# In[ ]:





# In[ ]:


dumpstr = 'mygoals_program' +  '_data.pkl'
with open(dumpstr, 'wb') as file:
    pkl.dump(mygoals, file)
file.close()


# In[ ]:




