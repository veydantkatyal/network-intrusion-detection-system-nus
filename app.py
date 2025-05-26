# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from scapy.all import sniff, IP, TCP, UDP  # Explicit protocol imports
from collections import defaultdict
import time
import threading

class PacketProcessor:
    """Process and analyze network packets"""
    def __init__(self):
        self.protocol_map = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
        self.packet_data = []
        self.start_time = time.time()
        self.lock = threading.Lock()

    def get_protocol_name(self, protocol_num):
        return self.protocol_map.get(protocol_num, f'OTHER({protocol_num})')

    def process_packet(self, packet):
        try:
            if packet.haslayer(IP):  # Updated layer check
                ip_layer = packet[IP]
                with self.lock:
                    packet_info = {
                        'timestamp': time.time(),
                        'source': ip_layer.src,
                        'destination': ip_layer.dst,
                        'protocol': self.get_protocol_name(ip_layer.proto),
                        'size': len(packet)
                    }
                    
                    # TCP/UDP detection
                    if packet.haslayer(TCP):
                        tcp_layer = packet[TCP]
                        packet_info.update({
                            'src_port': tcp_layer.sport,
                            'dst_port': tcp_layer.dport
                        })
                    elif packet.haslayer(UDP):
                        udp_layer = packet[UDP]
                        packet_info.update({
                            'src_port': udp_layer.sport,
                            'dst_port': udp_layer.dport
                        })
                    
                    self.packet_data.append(packet_info)
                    if len(self.packet_data) > 1000:
                        self.packet_data.pop(0)
        except Exception as e:
            pass

    def get_dataframe(self):
        with self.lock:
            return pd.DataFrame(self.packet_data)

def start_packet_capture():
    processor = PacketProcessor()
    def capture_packets():
        sniff(prn=processor.process_packet, store=False)
    threading.Thread(target=capture_packets, daemon=True).start()
    return processor

def main():
    st.set_page_config(page_title="Network Monitor", layout="wide")
    st.title("Real-time Network Traffic Dashboard")
    
    if 'processor' not in st.session_state:
        st.session_state.processor = start_packet_capture()
    
    df = st.session_state.processor.get_dataframe()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Packets", len(df))
    with col2:
        st.metric("Unique Sources", df['source'].nunique() if not df.empty else 0)
    with col3:
        st.metric("Data Volume", f"{df['size'].sum()/1024:.2f} KB" if not df.empty else "0 KB")
    
    # Visualizations
    if not df.empty:
        st.subheader("Protocol Distribution")
        protocol_counts = df['protocol'].value_counts()
        fig1 = px.pie(protocol_counts, names=protocol_counts.index, values=protocol_counts.values)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Traffic Over Time")
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
        time_counts = df.resample('2s', on='time').size()
        fig2 = px.line(time_counts, title="Packets per 2 Seconds")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Auto-refresh every 2 seconds
    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    # Quick protocol test
    test_packet = IP()/TCP(dport=80)
    assert test_packet.haslayer(IP) and test_packet.haslayer(TCP), "Protocol test failed!"
    main()
