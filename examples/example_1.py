from wakapy import User

user = User('/home/surister/data.json')
chart = user.pie_chart('lan')
# lan = languages. See the different options here.
chart.show()
# Shows the chart.
chart.save('/home/surister/mychart.png')
# Saves the chart to the desired filepath.
