import cv2
import urllib.request
import numpy as np

class iPhoneCamera:
    """
    Connect to iPhone camera via HTTP stream
    No driver needed!
    """
    
    def __init__(self, ip_address="10.235.95.196:8081"):
        """
        Args:
            ip_address: IP shown in iPhone app (e.g., "10.235.95.196:8081")
        """
        self.stream_url = f"http://{ip_address}/video"
        self.cap = None
    
    def connect(self):
        """Connect to iPhone camera stream"""
        print(f"📱 Connecting to iPhone at {self.stream_url}...")
        
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            
            if self.cap.isOpened():
                print("✅ Connected to iPhone camera!")
                return True
            else:
                print("❌ Could not connect. Check:")
                print("   1. iPhone and computer on same WiFi")
                print("   2. IP address is correct")
                print("   3. App is running on iPhone")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def read_frame(self):
        """Read frame from iPhone"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release connection"""
        if self.cap:
            self.cap.release()
        print("📱 iPhone camera disconnected")


# Usage
if __name__ == "__main__":
    # Replace with YOUR iPhone's IP address from the app
    iphone = iPhoneCamera(ip_address="10.235.95.196:8081")
    
    if iphone.connect():
        print("Press 'q' to quit\n")
        
        while True:
            frame = iphone.read_frame()
            
            if frame is not None:
                cv2.imshow('iPhone Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        iphone.release()
        cv2.destroyAllWindows()
    else:
        print("\n💡 Troubleshooting:")
        print("1. Make sure iPhone app is running")
        print("2. Check IP address matches what's shown in app")
        print("3. Ensure both devices on same WiFi network")