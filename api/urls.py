from django.urls import path
import api.views as views


app_name = 'api'

urlpatterns = [
    path('get_prediction/', views.Bot.as_view(), name='get-prediction'),
]
