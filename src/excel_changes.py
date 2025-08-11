import pandas as pd

#this code is just to get the healthy info from the excel sheet

df = pd.read_excel('data/Demographics_of_the_participants.xlsx')
df.columns = df.columns.str.strip()
df_healthy  = df[df['Diagnosis'].str.lower() == 'healthy']
df_healthy = df_healthy.rename(columns={'Image ID':'Filename'})
df_output = df_healthy[['Filename','Age']]
df_output.to_excel('data/metadata_healthy_only.xlsx')
print('New file donne : data/metadatahealthy_only.xlsx')
