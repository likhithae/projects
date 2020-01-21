import datetime

from crispy_forms.helper import FormHelper
from django.forms import ModelForm, DateInput, TimeInput,  forms
from home.models import labAppointment

class labAppointmentForm(ModelForm):
    class Meta:
        model = labAppointment
        fields = ['date', 'time']
        widgets = {
            'date': DateInput(attrs={'type': 'date'}),
            'time': TimeInput(attrs={'type': 'time'}),
        }

        helper = FormHelper()
        helper.form_method = 'POST'


    def clean_date(self):
        date = self.cleaned_data['date']
        if date < datetime.date.today():
            raise forms.ValidationError("Sorry, Invalid Date!")
        return date

