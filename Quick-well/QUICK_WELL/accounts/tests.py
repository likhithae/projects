# from django.test import TestCase, Client
#
# class UrlTestingsignup(TestCase):
#     def test_accounts(self):
#         self.client = Client()
#         response = self.client.get('/signup4/')
#         # self.assertRedirects(response,expected_url="")
#         self.assertEqual(response.status_code,200)
#
# class UrlTestinglogin(TestCase):
#     def test_accounts(self):
#         self.client = Client()
#         response = self.client.get('/login/')
#         self.assertEqual(response.status_code,200)
#
# class UrlTestingchangepassword(TestCase):
#     def test_accounts(self):
#         self.client = Client()
#         response = self.client.get('/login/changepassword/')
#         self.assertRedirects(response,expected_url="/login/")
#
# class UrlTestinglogout(TestCase):
#     def test_accounts(self):
#         self.client = Client()
#         response = self.client.get('/logout/')
#         self.assertEqual(response.status_code,301)
#
# class UrlTestinghome(TestCase):
#     def test_accounts(self):
#         self.client = Client()
#         response = self.client.get('/login/home/')
#         self.assertRedirects(response,expected_url="/login/")
# #
# # class UrlTestinghome(TestCase):
# #     def test_accounts(self):
# #         self.client = Client()
# #         response = self.client.get('/login/home/')
# #         self.assertEqual(response.status_code,200)
#
# class UrlTestingcontact(TestCase):
#     def test_accounts(self):
#         self.client = Client()
#         response = self.client.get('/login/contact/')
#         self.assertRedirects(response,expected_url="/login")
#
# class UrlTestingtest(TestCase):
#     def test_accounts(self):
#         self.client = Client()
#         response = self.client.get('/login/test/')
#         self.assertRedirects(response,expected_url="/login/")
#
# class UrlTestinghome(TestCase):
#     def test_accounts(self):
#         self.client = Client()
#         response = self.client.get('/login/doctor_update/')
#         self.assertRedirects(response,expected_url="/login")
