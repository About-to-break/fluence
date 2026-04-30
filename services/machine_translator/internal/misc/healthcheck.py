import torch.cuda
import logging

def healthcheck() -> bool:
    try:
        if not torch.cuda.is_available():
            raise Exception('No GPU available')

        return True

    except Exception as e:
        logging.error(e)
        return False

if __name__ == '__main__':
    print(healthcheck())
