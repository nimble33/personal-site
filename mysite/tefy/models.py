from django.db import models


class Post(models.Model):

    text = models.TextField()


    # def publish(self):
    #     self.published_date = timezone.now()
    #     self.save()

    # def __str__(self):
    #     return self.title