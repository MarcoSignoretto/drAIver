class MotorProtocol:

    # most significative bit is the motor identifier ( 1 is left, 0 is right)
    MOTOR_LEFT = 0x8000
    MOTOR_RIGHT = 0x0000

    # less significative bit of the most significative byte is the motor direction (1 back, 0 forth)
    #MOTOR_BACK = 0x0100
    #MOTOR_FORTH = 0x0000

    MOTOR_SPEED_MASK = 0x01FF  # get also the direction

    COMMUNICATION_MASK = 0xFFFFFFFF  # 4 bytes
    COMMUNICATION_PACKET_SIZE = 4  # 4 bytes in communication
    MOTOR_PACKET_MASK = 0x0000FFFF

    def pack(self, motor, speed):
        """
        pack(motor, speed)

            Prepare a packet for motor commands

            Parameters
            ----------
            motor : int
                Motor identifier ( MotorProtocol.MOTOR_LEFT or MotorProtocol.MOTOR_RIGHT
            speed : int
                Motor speed a value between -255 and 255, negative means back direction, positive is forth direction
            Returns
            -------
            out : int
                The packet for a single motor

            Examples
            --------

            pack = MotorProtocol.pack(MOTOR_LEFT, 100)
        """

        packet = speed & self.MOTOR_SPEED_MASK
        packet = packet ^ motor
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

        packet = motor_left_packet << 16
        packet = packet ^ motor_right_packet
        packet = packet & self.COMMUNICATION_MASK
        return packet

    def split(self, packet):

        right_packet = packet & self.MOTOR_PACKET_MASK
        left_packet = (packet >> 16) & self.MOTOR_PACKET_MASK
        return left_packet, right_packet

    def decompose(self, packet):
        speed = (packet & self.MOTOR_SPEED_MASK)
        motor = (packet >> 15) & 0x0001  # keep most significative bit
        return motor, speed




