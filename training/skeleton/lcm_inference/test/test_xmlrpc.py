from multiprocessing import Process, Queue
from threading import Thread
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

def serve_wrapper(my_funcs):
    # Create server
    with SimpleXMLRPCServer(('localhost', 8000),
                            requestHandler=RequestHandler) as server:
        server.register_introspection_functions()

        # Register pow() function; this will use the value of
        # pow.__name__ as the name, which is just 'pow'.
        server.register_function(pow)

        # Register a function under a different name
        def adder_function(x, y):
            return x + y
        server.register_function(adder_function, 'add')

        # Register an instance; all the methods of the instance are
        # published as XML-RPC methods (in this case, just 'mul').

        # server.register_instance(MyFuncs())
        server.register_instance(my_funcs)

        # Run the server's main loop
        server.serve_forever()

if __name__ == "__main__":
    class MyFuncs:
        def __init__(self):
            self.test_str = 'start_str'

        def mul(self, x, y):
            return x * y
        
        def return_string(self):
            return self.test_str
    my_funcs = MyFuncs()
    q1 = Queue()
    p1 = Process(target=serve_wrapper, args=(my_funcs,))
    p1.daemon = True
    p1.start()
    # thread = Thread(target=serve_wrapper)
    # thread.daemon = True
    # thread.start()

    import time
    time.sleep(4.0)
    print('4 secs have passed')

    my_funcs.test_str = 'end_str'

    from IPython import embed
    embed()