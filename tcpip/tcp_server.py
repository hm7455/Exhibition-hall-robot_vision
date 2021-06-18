import socket

tcp_server_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

tcp_server_socket.bind(("192.168.1.9",9090)) #绑定本地ip 端口为9090

tcp_server_socket.listen(5)

client_socket= tcp_server_socket.accept()

print(client_socket)

clientAddr=client_socket[1]
client_socket=client_socket[0]

recv_data=client_socket.recv(1024)
print('接收到的数据为:', recv_data.decode('gbk'))

client_socket.send("thank you !".encode('gbk'))

client_socket.close()

#客户端发送数据给服务器


