from time import sleep
import multiprocessing
from alphadev.service import Program, RPCService, RPCClient

class Internal:
    def __init__(self, attr):
        self.attr = attr
        
    @staticmethod
    def static_plus1(arg):
        return arg + 1
    
    def plus1(self, arg):
        return arg + 1
    
    def plus_attr(self, arg):
        return arg + self.attr
    
    def __call__(self, arg):
        return arg + 1

def runner(client: RPCClient):
    print('runner sleeps 5')
    sleep(5)
    assert client.static_plus1(1) == 2
    assert client.plus1(1) == 2
    assert client.plus_attr(1) == 11
    assert client(1) == 2

redis_config = {
    'type': 'redis',
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

def test_program():
    prog = Program()
    internal_factory = lambda: Internal(10)
    with prog.group('test'):
        client = prog.add_service(RPCService(instance_factory=internal_factory, conn_config=redis_config))
    worker = multiprocessing.Process(target=runner, args=(client,))
    worker.start()
    prog.launch()
    worker.join()
    prog.stop()
    assert worker.exitcode == 0
    
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    test_program()
    print('test_program passed')