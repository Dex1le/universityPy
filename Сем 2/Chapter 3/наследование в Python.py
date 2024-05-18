from typing import List


class Contact:
    all_contacts: List["Contact"] = []

    def __init__(self, /,  name: str = '', email: str = '', **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.email = email
        self.all_contacts.append(self)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.name!r}, {self.email!r}"
            f")"
        )


class Supplier(Contact):
    def order(self, order: "Order") -> None:
        print(
            f"'{order}' order to '{self.name}'"
        )


class AdressHolder:
    def __init__(self,
                 /,
                 street: str = "",
                 city: str = "",
                 state: str = "",
                 code: str = "",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.street = street
        self.city = city
        self.state = state
        self.code = code


class Friend(Contact, AdressHolder):
    def __init__(
            self,
            /,
            phone: str = "",
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.phone = phone


class Emailable:
    email: str


f = Friend(89626911156, name='Slava', email='opewjwpeo@gmail', street='Kolasa',
           city='Kolasa', state='Kolasa', code='Kolasa')
s = Supplier('Sup Sup', 'somebody')
cont = Contact('Ilya', 'ofkewpofk')
cont1 = Contact('Bro', 'euwoifjwe')

print(Contact.all_contacts)


# class MailSender(Emailable):
#     def send_mail(self, message: str) -> None:
#         print(f"Sending mail to {self.email}")


# class EmailableContact(Contact, MailSender):
#     pass


# m = MailSender()
# m.send_mail()