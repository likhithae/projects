import datetime
from crispy_forms.helper import FormHelper
from django.forms import ModelForm, DateInput, Select, TextInput, NumberInput, forms,FileInput
from home.models import fundraiser


class fundraiserForm(ModelForm):
    class Meta:
        category = [
            ('medical','Medical'),
            ('ngo','NGO'),
            ('hospital','Hospital'),
            ('clinic','Clinic'),
        ]

        model = fundraiser
        fields = ['category','Title','goal_amount','beneficiary_name','beneficiary_relation','Fundraiser_story','End_date', 'photo', 'account_number', 'accountholder_name', 'ifsc_code']
        widgets = {
            'category':Select(choices=category),
            'Title':TextInput(attrs={'type': 'text'}),
            'goal_amount':NumberInput(attrs={'type': 'float'}),
            'beneficiary_name': TextInput(attrs={'type': 'text'}),
            'beneficiary_relation': TextInput(attrs={'type': 'text'}),
            'Fundraiser_story': TextInput(attrs={'type': 'text'}),
            'End_date': DateInput(attrs={'type': 'date'}),
            #'account_number': NumberInput(attrs={'type': 'int'}),
            'accountholder_name': TextInput(attrs={'type': 'text'}),
            'ifsc_code': TextInput(attrs={'type': 'text'}),
        }

        helper = FormHelper()
        helper.form_method = 'POST'

    def clean_date(self):
        date = self.cleaned_data['End_date']
        if date < datetime.date.today():
            raise forms.ValidationError("Sorry, Invalid Date!")
        return date


# class DonateForm(ModelForm):
#     class Meta:
#         model = Donar
#         fields = ['name','email_id', 'Contribution_amount','comment','Creditcard_no','Cardholdername','expiry_date','cvv_no','city']
#         widgets = {
#             'name':TextInput(attrs={'type':'text'}),
#             'email_id': TextInput(attrs={'type': 'text'}),
#             'Contribution_amount':NumberInput(attrs={'type': 'float'}),
#             'comment':TextInput(attrs={'type': 'text'}),
#             'Creditcard_no':NumberInput(attrs={'type': 'float'}),
#             'Cardholdername': TextInput(attrs={'type': 'text'}),
#             'expiry_date': TextInput(attrs={'type': 'text'}),
#             'cvv_no': TextInput(attrs={'type': 'text'}),
#             'city': TextInput(attrs={'type': 'text'}),
#         }
#
#         helper = FormHelper()
#         helper.form_method = 'POST'

#