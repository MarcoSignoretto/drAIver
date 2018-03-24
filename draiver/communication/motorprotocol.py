import numpy as np


class MotorProtocol:


    # TODO improve protocol use only 2 bytes with speed only range -100 +100

    MOTOR_SPEED_MASK = 0xFF  # get also the direction

    COMMUNICATION_MASK = 0xFFFF  # 2 bytes
    COMMUNICATION_PACKET_SIZE = 2  # 2 bytes in communication
    MOTOR_PACKET_MASK = 0x00FF

    def pack(self, speed):
        """
        pack(motor, speed)

            Prepare a packet for motor commands

            Parameters
            ----------
            speed : int
                Motor speed a value between -100 and 100, negative means back direction, positive is forth direction
            Returns
            -------
            out : int
                The packet for a single motor

            Examples
            --------

            pack = MotorProtocol.pack(100)
        """
        speed = int(np.round(speed))
        packet = speed & self.MOTOR_SPEED_MASK
        return packet

    def merge(self, motor_left_packet, motor_right_packet):
        """
        merge(motor, action, speed)

            Prepare a packet for motor commands

            Parameters
            ----------
            motor_left_packet : int
                Packet for the left motor
            motor_right_packet : int
                Packet for the right motor
            Returns
            -------
            out : int
                The communication packet for both motors

            Examples
            --------

            pack = MotorProtocol.merge(MotorProtocol.pack(MOTOR_LEFT, 100), MotorProtocol.pack(MOTOR_RIGHT, 100))
        """

        packet = motor_left_packet << 8
        packet = packet ^ motor_right_packet
        packet = packet & self.COMMUNICATION_MASK
        return packet

    def split(self, packet):

        right_packet = packet & self.MOTOR_PACKET_MASK
        left_packet = (packet >> 8) & self.MOTOR_PACKET_MASK
        return left_packet, right_packet




