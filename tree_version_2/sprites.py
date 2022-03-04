import pygame


class AnimeObject(pygame.sprite.Sprite):
    """
    Visualize operations through pygame

    """

    def __init__(self, frames_paths, fps=5):
        super().__init__()
        self.frames = []
        for frame in frames_paths:
            self.frames.append(pygame.image.load(frame).convert_alpha())
        self._frames_len = len(self.frames)
        self.image = self.frames[0]
        self.rect = self.image.get_rect()
        self.current_frame = 0
        self._period = 1000. / fps
        self._frequency = fps / 1000.
        self._start_time = 0
        self._next_update = 0

    def update(self, dt, t):
        # simple way, but mostly slower
        # self.current_frame = (int((t - self._start_time) * self._frequency)) % self._frames_len
        # self.image = self.frames[self.current_frame]
        # self.rect = self.image.get_rect(center=self.rect.center)
        # more efficient for low fps
        if t >= self._next_update:
            delta = t - self._next_update
            skipped_frames = int(self._frequency * delta)
            self._next_update += (skipped_frames+1) * self._period
            self.current_frame += 1 + skipped_frames
            self.current_frame %= self._frames_len
            self.image = self.frames[self.current_frame]
            self.rect = self.image.get_rect(center=self.rect.center)

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def set_pos(self, pos):
        self.rect.x, self.rect.y = pos[0], pos[1]

    def move(self, x_step, y_step):
        self.rect.move_ip(x_step, y_step)

    def set_frames(self, frames):
        self.frames.clear()
        if isinstance(frames[0], str):
            for frame in frames:
                f = pygame.image.load(frame).convert_alpha()
                f = pygame.transform.scale(f, (self.rect.width, self.rect.height))
                self.frames.append(f)
        else:
            for frame in frames:
                self.frames.append(pygame.transform.scale(frame, (self.rect.width, self.rect.height)))
        self.image = self.frames[0]
        self._frames_len = len(self.frames)

    def resize(self, width, height):
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(center=self.rect.center)
        for i, frame in enumerate(self.frames):
            self.frames[i] = pygame.transform.scale(frame, (width, height))


class StaticObject(pygame.sprite.Sprite):
    def __init__(self, image_path):
        super().__init__()
        self.image = pygame.image.load(image_path)
        self.rect = self.image.get_rect()

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def resize(self, width, height):
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(center=self.rect.center)

    def set_pos(self, pos):
        self.rect.x, self.rect.y = pos[0], pos[1]

    def set_image(self, image):
        if isinstance(image, str):
            self.image = pygame.image.load(image).convert_alpha()
        else:
            self.image = image


class RigidBody(StaticObject):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.speed = 0.0
        self.speed_x, self.speed_y = 0.0, 0.0
        self.acceleration = 0.0
        self.start_pos = (0, 0)
        self.end_pos = (0, 0)
        self._image_copy = self.image.copy()

    def smooth_slide(self, dt):
        if abs(self.rect.center[0] - self.start_pos[0]) < abs(self.start_pos[0] - self.end_pos[0]) / 2 or \
                abs(self.rect.center[1] - self.start_pos[1]) < abs(self.start_pos[1] - self.end_pos[1]) / 2:
            offset_x = self.speed_x * dt
            offset_y = self.speed_y * dt
            hypotenuse = ((
                        (self.end_pos[0] - self.start_pos[0]) ** 2 + (self.end_pos[1] - self.start_pos[1]) ** 2) ** 0.5)
            self.speed += self.acceleration * dt
            self.speed_x = self.speed * (self.end_pos[0] - self.start_pos[0]) / hypotenuse
            self.speed_y = self.speed * (self.end_pos[1] - self.start_pos[1]) / hypotenuse
            self.rect = self._image_copy.get_rect(center=self.rect.center)
            self.rect.move_ip(round(offset_x), round(offset_y))
            # change size
            speed_rate = abs(self.speed)
            self.image = pygame.transform.scale(self._image_copy, (int(speed_rate * 60) + 5, int(speed_rate * 60) + 5))
            self.rect = self.image.get_rect(center=self.rect.center)
        else:
            offset_x = self.speed_x * dt
            offset_y = self.speed_y * dt
            hypotenuse = ((
                        (self.end_pos[0] - self.start_pos[0]) ** 2 + (self.end_pos[1] - self.start_pos[1]) ** 2) ** 0.5)
            self.speed += -self.acceleration * dt
            self.speed_x = self.speed * (self.end_pos[0] - self.start_pos[0]) / hypotenuse
            self.speed_y = self.speed * (self.end_pos[1] - self.start_pos[1]) / hypotenuse
            self.rect = self._image_copy.get_rect(center=self.rect.center)
            self.rect.move_ip(round(offset_x), round(offset_y))
            # change size
            speed_rate = abs(self.speed)
            self.image = pygame.transform.scale(self._image_copy, (int(speed_rate * 60) + 5, int(speed_rate * 60) + 5))
            self.rect = self.image.get_rect(center=self.rect.center)

    def set_speed(self, speed):
        self.speed = speed

    def set_acceleration(self, accel):
        self.acceleration = accel

    def set_start_end_pos(self, start, end):
        self.start_pos = start
        self.end_pos = end


class Tree(AnimeObject):
    def __init__(self, frames_paths, fps=5):
        super().__init__(frames_paths, fps)
        self.is_chopped = False  # if this tree is chopped, set True, otherwise, set False
        self.age = 0.0  # the age of this tree
        self.maximal_age = 7.0  # we assume that trees won't grow anymore when they reach a certain age
        self.timber_value = {
            '-1.0': 0.0,
            '0.0': 0.0,
            '1.0': 1.0,
            '2.0': 3.0,
            '3.0': 6.0,
            '4.0': 10.0,
            '5.0': 15.0,
            '6.0': 23.0,
            '7.0': 30.0
        }  # the value(selling price) of this tree wrt. age

    def get_timber_value(self, age):
        if isinstance(age, str):
            return self.timber_value[age]
        else:
            return self.timber_value[str(age)]
