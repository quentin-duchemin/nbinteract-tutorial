from datetime import date

from wakapy import User

date1 = '2018-10-10'
date2 = date(year=2018, month=10, day=17)
# Dates can either be a str or a datetime.date object.

a = User('/home/surister/info.json')

a_slice = a.get_slice(date1, date2)
# Returns a Slice object containing
# the Days object between the two given dates

chart = a.pie_chart('proj', num=3, sliced=True)
# Num is the number of projets that will be displayed
# Sliced is set to True, so the chart will be created with
# sliced object created before.

chart.save('/home/surister/mychart.png')
