from utils import *

def get_simbol(state):
  try:
    return states_meta_data_df[states_meta_data_df['name'] == state]['state'].iloc[0]
  except:
    return None
    

def get_lng(state):
  return states_meta_data_df[states_meta_data_df['name'] == state]['longitude'].iloc[0]


def get_lat(state):
  return states_meta_data_df[states_meta_data_df['name'] == state]['latitude'].iloc[0]


def get_state_code(state, states_df):
  try:
    return states_df[states_df['State'] == state]['Code'].iloc[0]
  except:
    return np.nan
    
# MAIN

# Load data:
covid_spread_state_df = pd.read_csv("drive/My Drive/Colab Notebooks/Big Data/final_project/Data/BGU DATA/Covid Spread/covid_spread_us.csv")
states_df = pd.read_csv("drive/My Drive/Colab Notebooks/Big Data/final_project/Data/BGU DATA/Covid Spread/us_states.csv")

# Data preprocessing:
covid_spread_state_df['state_code'] = covid_spread_state_df['state'].apply(lambda x: get_state_code(state=x, states_df=states_df))  
covid_spread_state_df = covid_spread_state_df.dropna(subset=['state_code'])
covid_spread_state_df['cases'].astype(int)
covid_spread_state_df['deaths'].astype(int)
covid_spread_state_df['datetime'] = covid_spread_state_df['date'].astype('datetime64[ns]')
covid_spread_state_df['week_num'] = covid_spread_state_df['datetime'].apply(lambda x: x.strftime("%V"))

# Compute cases per day:
cases = []
deaths = []
counter = 0
for i,row in covid_spread_state_df.iterrows():
  case = row['cases']
  death = row['deaths']
  state = row['state']
  if counter==0:
    cases.append(case)
    deaths.append(death)
  else:
    df_tmp = covid_spread_state_df[:counter]
    df_state_tmp = df_tmp[df_tmp['state']==state][-1:]
    if len(df_state_tmp)==0:
      cases.append(case)
      deaths.append(death)
    else:
      last_cases = df_state_tmp.iloc[0]['cases']
      last_death = df_state_tmp.iloc[0]['deaths']
      cases.append(case - last_cases)
      d = death - last_death
      if d<0:
        d= d = 0
      deaths.append(d)
  counter +=1
covid_spread_state_df['cases_per_day'] = cases
covid_spread_state_df['deaths_per_day'] = deaths
covid_spread_state_df = covid_spread_state_df.rename(columns={"cases_per_day": 'covid cases', "deaths_per_day": 'covid deaths'})

# Get Coronavirus daily spread:
corona_per_dates_gb = covid_spread_state_df.groupby(by="date").sum().reset_index()
t = corona_per_dates_gb['date']
cases = corona_per_dates_gb['covid cases']
deaths = corona_per_dates_gb['covid deaths']

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=t, y=savgol_filter(deaths,21,5), name="Deaths", line=dict(color='red', width=2) ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=t, y=savgol_filter(cases,11,3), name="Cases", line=dict(color='royalblue', width=2)),
    secondary_y=True,
)

fig.update_layout(
    title_text="Coronavirus daily spread in USA"
)
fig.update_xaxes(title_text="Day")
fig.update_yaxes(title_text="<b>Covid-19 Deaths</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Covid-19 Cases</b>", secondary_y=True)
fig.show()

# USA covid tweets sentiment:
covid_spread_state_gb = covid_spread_state_df.groupby(by=['date','state_code']).sum().reset_index()
covid_spread_state_gb = covid_spread_state_gb.rename(columns={'state_code': 'State_abbv'})
px.choropleth(covid_spread_state_gb, locations='State_abbv',
              locationmode='USA-states', scope="usa", color='cases',animation_frame="date", color_continuous_scale="Reds", title="Covid cases spread in USA states")

# Save dfs:
output_url = "drive/My Drive/Colab Notebooks/Big Data/final_project/Output/"
write_data(covid_spread_state_df, url=output_url+"covid_spread_state_df.csv")
write_data(covid_spread_state_gb, url=output_url+"covid_spread_state_gb.csv")
write_data(corona_per_dates_gb, url=output_url+"corona_per_dates_gb.csv")
