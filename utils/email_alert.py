import smtplib
import time
import cv2
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import streamlit as st


def send_email(receiver_email, frame, location="Unknown", timestamp="Unknown", confidence=0.0, max_retries=3, delay=5):
    """Send an email alert with fire detection details and a snapshot.

    Args:
        receiver_email: Recipient email address.
        frame: OpenCV frame (BGR) to attach as a snapshot.
        location: Location string where fire was detected.
        timestamp: Detection timestamp string.
        confidence: Detection confidence percentage.
        max_retries: Number of retry attempts on failure.
        delay: Seconds to wait between retries.

    Returns:
        True if email was sent successfully, False otherwise.
    """
    sender_email = st.secrets.get("SENDER_EMAIL", "sudhimallaavinash00@gmail.com")
    sender_password = st.secrets.get("SENDER_PASSWORD", "qlpu gzfj ldlu abos")
    attempt = 0
    server = None

    while attempt < max_retries:
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = 'FIRE ALERT - Immediate Attention Required'

            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; color: #333;">
                <h2 style="color: #D32F2F;">Fire Detected</h2>
                <p>The AI Fire Detection System has detected a fire. Details below:</p>
                <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
                    <tr><td style="padding: 8px; font-weight: bold;">Timestamp</td>
                        <td style="padding: 8px;">{timestamp}</td></tr>
                    <tr style="background: #f9f9f9;">
                        <td style="padding: 8px; font-weight: bold;">Location</td>
                        <td style="padding: 8px;">{location}</td></tr>
                    <tr><td style="padding: 8px; font-weight: bold;">Confidence</td>
                        <td style="padding: 8px;">{confidence:.1f}%</td></tr>
                </table>
                <p style="margin-top: 16px;">A snapshot of the detected fire is attached.</p>
                <hr>
                <p style="font-size: 12px; color: #999;">
                    This is an automated alert from the AI Wildfire Detection System.
                </p>
            </body>
            </html>
            """
            msg.attach(MIMEText(html_body, 'html'))

            _, buffer = cv2.imencode('.jpg', frame)
            img_data = buffer.tobytes()
            img = MIMEImage(img_data, name="Fire.jpg")
            img.add_header('Content-Disposition', 'attachment', filename="Fire.jpg")
            msg.attach(img)

            server.send_message(msg)
            break

        except Exception as e:
            attempt += 1
            time.sleep(delay)

        finally:
            try:
                if server:
                    server.quit()
            except Exception:
                pass

    return attempt < max_retries
