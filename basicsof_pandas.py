from pandas import DataFrame, read_csv

# General syntax to import a library but no functions: 
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number
print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)
Python version 3.7.13 (default, Mar 16 2022, 17:37:17) 
[GCC 7.5.0]
Pandas version 1.3.5
Matplotlib version 3.2.2
# The initial set of baby names and birth rates
names = ['Abhishek','Anandi','Siddharth','Parth','Deepak']
births = [968, 50, 180, 578, 973]
BabyDataSet = list(zip(names,births))
BabyDataSet
[('Abhishek', 968),
 ('Anandi', 50),
 ('Siddharth', 180),
 ('Parth', 578),
 ('Deepak', 973)]
df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])
df
Names	Births
0	Abhishek	968
1	Anandi	50
2	Siddharth	180
3	Parth	578
4	Deepak	973
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
df = pd.read_csv("/content/drive/MyDrive/Machine Learning (Cllg)/statewisetestingsamples.csv")
Df1=pd.read_csv("/content/drive/MyDrive/Machine Learning (Cllg)/Salary_Data.csv")
df.set_index("TotalSamples")
Df1
YearsExperience	Salary
0	1.1	39343.0
1	1.3	46205.0
2	1.5	37731.0
3	2.0	43525.0
4	2.2	39891.0
5	2.9	56642.0
6	3.0	60150.0
7	3.2	54445.0
8	3.2	64445.0
9	3.7	57189.0
10	3.9	63218.0
11	4.0	55794.0
12	4.0	56957.0
13	4.1	57081.0
14	4.5	61111.0
15	4.9	67938.0
16	5.1	66029.0
17	5.3	83088.0
18	5.9	81363.0
19	6.0	93940.0
20	6.8	91738.0
21	7.1	98273.0
22	7.9	101302.0
23	8.2	113812.0
24	8.7	109431.0
25	9.0	105582.0
26	9.5	116969.0
27	9.6	112635.0
28	10.3	122391.0
29	10.5	121872.0
Df2=pd.read_csv("/content/drive/MyDrive/Machine Learning (Cllg)/diabetes.csv")
Df2
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
0	6	148	72	35	0	33.6	0.627	50	1
1	1	85	66	29	0	26.6	0.351	31	0
2	8	183	64	0	0	23.3	0.672	32	1
3	1	89	66	23	94	28.1	0.167	21	0
4	0	137	40	35	168	43.1	2.288	33	1
...	...	...	...	...	...	...	...	...	...
763	10	101	76	48	180	32.9	0.171	63	0
764	2	122	70	27	0	36.8	0.340	27	0
765	5	121	72	23	112	26.2	0.245	30	0
766	1	126	60	0	0	30.1	0.349	47	1
767	1	93	70	31	0	30.4	0.315	23	0
768 rows × 9 columns

from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Df2.dtypes
Pregnancies                   int64
Glucose                       int64
BloodPressure                 int64
SkinThickness                 int64
Insulin                       int64
BMI                         float64
DiabetesPedigreeFunction    float64
Age                           int64
Outcome                       int64
dtype: object
Df1.sort_values(['Salary'], ascending=False)
YearsExperience	Salary
28	10.3	122391.0
29	10.5	121872.0
26	9.5	116969.0
23	8.2	113812.0
27	9.6	112635.0
24	8.7	109431.0
25	9.0	105582.0
22	7.9	101302.0
21	7.1	98273.0
19	6.0	93940.0
20	6.8	91738.0
17	5.3	83088.0
18	5.9	81363.0
15	4.9	67938.0
16	5.1	66029.0
8	3.2	64445.0
10	3.9	63218.0
14	4.5	61111.0
6	3.0	60150.0
9	3.7	57189.0
13	4.1	57081.0
12	4.0	56957.0
5	2.9	56642.0
11	4.0	55794.0
7	3.2	54445.0
1	1.3	46205.0
3	2.0	43525.0
4	2.2	39891.0
0	1.1	39343.0
2	1.5	37731.0
Sorted = df.sort_values(['Negative'], ascending=False)
Sorted.head(50)
Date	State	TotalSamples	Negative	Positive
1546	06/06/2020	Tamil Nadu	576695	550643.0	30152.0
1545	05/06/2020	Tamil Nadu	560673	535254.0	28694.0
1544	04/06/2020	Tamil Nadu	544981	517137.0	27256.0
1543	03/06/2020	Tamil Nadu	528534	502173.0	25872.0
1542	02/06/2020	Tamil Nadu	514433	489258.0	24586.0
1541	01/06/2020	Tamil Nadu	503339	479208.0	23495.0
1540	31/05/2020	Tamil Nadu	491962	468940.0	22333.0
1452	05/06/2020	Rajasthan	480910	465349.0	10084.0
1539	30/05/2020	Tamil Nadu	479155	457405.0	21184.0
1451	04/06/2020	Rajasthan	467129	451826.0	9862.0
1049	06/06/2020	Maharashtra	538009	451764.0	80229.0
1538	29/05/2020	Tamil Nadu	466550	445668.0	20246.0
1450	03/06/2020	Rajasthan	454788	440850.0	9652.0
1048	05/06/2020	Maharashtra	524002	440445.0	77793.0
1537	28/05/2020	Tamil Nadu	455216	435279.0	19372.0
63	06/06/2020	Andhra Pradesh	436335	431875.0	3588.0
1047	04/06/2020	Maharashtra	511136	430100.0	74860.0
1449	02/06/2020	Rajasthan	440789	428471.0	9373.0
1536	27/05/2020	Tamil Nadu	442970	423775.0	18545.0
1046	03/06/2020	Maharashtra	498577	420644.0	72300.0
62	05/06/2020	Andhra Pradesh	423564	419314.0	3427.0
1535	26/05/2020	Tamil Nadu	431739	413455.0	17728.0
1448	01/06/2020	Rajasthan	425184	411965.0	9100.0
61	04/06/2020	Andhra Pradesh	413733	409621.0	3377.0
1045	02/06/2020	Maharashtra	484784	409178.0	70013.0
1534	25/05/2020	Tamil Nadu	421450	403762.0	17082.0
60	03/06/2020	Andhra Pradesh	403747	399776.0	3279.0
1044	01/06/2020	Maharashtra	472344	399419.0	67655.0
1447	31/05/2020	Rajasthan	409777	396789.0	8831.0
1533	24/05/2020	Tamil Nadu	409615	392690.0	16277.0
1043	31/05/2020	Maharashtra	463177	392516.0	65168.0
59	02/06/2020	Andhra Pradesh	395681	391890.0	3200.0
1446	30/05/2020	Rajasthan	395490	382315.0	8617.0
1532	23/05/2020	Tamil Nadu	397340	381216.0	15512.0
1042	30/05/2020	Maharashtra	448661	380425.0	62228.0
58	01/06/2020	Andhra Pradesh	383315	379639.0	3118.0
1531	22/05/2020	Tamil Nadu	385185	369929.0	14735.0
1041	29/05/2020	Maharashtra	434565	369442.0	59546.0
57	31/05/2020	Andhra Pradesh	372748	369177.0	3045.0
1445	29/05/2020	Rajasthan	379315	365925.0	8365.0
56	30/05/2020	Andhra Pradesh	363378	359917.0	2944.0
1040	28/05/2020	Maharashtra	420473	358253.0	56948.0
1530	21/05/2020	Tamil Nadu	372532	357898.0	13967.0
1444	28/05/2020	Rajasthan	365556	351861.0	8067.0
55	29/05/2020	Andhra Pradesh	353874	350544.0	2874.0
833	05/06/2020	Karnataka	360720	349951.0	4835.0
1529	20/05/2020	Tamil Nadu	360068	346311.0	13191.0
1039	27/05/2020	Maharashtra	405020	345151.0	54758.0
54	28/05/2020	Andhra Pradesh	342236	338991.0	2841.0
1443	27/05/2020	Rajasthan	350600	338611.0	7816.0
df.loc[56]
Date                30/05/2020
State           Andhra Pradesh
TotalSamples            363378
Negative              359917.0
Positive                2944.0
Name: 56, dtype: object
df.rank()
Date	State	TotalSamples	Negative	Positive
0	927.5	3.5	239.0	150.0	301.0
1	1346.0	3.5	340.0	NaN	382.5
2	1520.5	3.5	350.0	NaN	419.5
3	18.5	3.5	417.0	NaN	419.5
4	897.5	3.5	579.5	NaN	419.5
...	...	...	...	...	...
1770	48.5	1744.5	1631.0	NaN	1576.0
1771	116.0	1744.5	1643.0	NaN	1584.0
1772	181.5	1744.5	1653.0	NaN	1594.0
1773	247.0	1744.5	1664.0	NaN	1603.0
1774	318.5	1744.5	1671.0	NaN	1614.0
1775 rows × 5 columns

df.head()
Date	State	TotalSamples	Negative	Positive
0	17/04/2020	Andaman and Nicobar Islands	1403	1210.0	12.0
1	24/04/2020	Andaman and Nicobar Islands	2679	NaN	27.0
2	27/04/2020	Andaman and Nicobar Islands	2848	NaN	33.0
3	01/05/2020	Andaman and Nicobar Islands	3754	NaN	33.0
4	16/05/2020	Andaman and Nicobar Islands	6677	NaN	33.0
gb=df.groupby("State")
