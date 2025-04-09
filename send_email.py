import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


def send_email_with_image(smtp_server, port, sender_email, password, receiver_email, subject, body, image_path):
    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Attach the body text to the email
    msg.attach(MIMEText(body, 'plain'))

    # Open the image file in binary mode and attach it to the email
    with open(image_path, 'rb') as img:
        image = MIMEImage(img.read())
        image.add_header('Content-ID', '<{}>'.format(image_path))
        msg.attach(image)

    # Connect to the SMTP server and send the email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()  # Use TLS for security
        server.login(sender_email, password)
        server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        server.quit()
