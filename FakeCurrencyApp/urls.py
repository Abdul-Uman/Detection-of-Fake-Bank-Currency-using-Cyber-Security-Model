from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('UserLoginAction', views.UserLoginAction, name="UserLoginAction"),	   
	       path('Signup', views.Signup, name="Signup"),
	       path('SignupAction', views.SignupAction, name="SignupAction"),
	       path('UploadDataset', views.UploadDataset, name="UploadDataset"),
	       path('UploadDatasetAction', views.UploadDatasetAction, name="UploadDatasetAction"),
	       path('TrainML', views.TrainML, name="TrainML"),
	       path('FakeDetection', views.FakeDetection, name="FakeDetection"),
	       path('FakeDetectionAction', views.FakeDetectionAction, name="FakeDetectionAction"),
	       path('PreprocessDataset', views.PreprocessDataset, name="PreprocessDataset"),
]