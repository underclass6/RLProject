import random
import sys

import os

import pygame.font

from sprites import *


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((600, 700))
    screen.fill((0, 0, 0))
    clock = pygame.time.Clock()

    # create ground
    ground_paths = [os.getcwd() + r'\assets\PixelTrees\ground' + str(i+1) + '.png' for i in range(4)]
    background = pygame.surface.Surface((600, 600))
    for i in range(10):
        for j in range(10):
            g = StaticObject(ground_paths[random.randint(0, 3)])
            g.resize(59, 59)
            g.set_pos((i * 60, j * 60))
            background.blit(g.image, g.rect)
    screen.blit(background, (0, 0))

    # create font of hint for the number of timber
    timber_value = 0.0  # value of single timber
    timber_profit = 0.0  # total profit
    timber_num_font = pygame.font.SysFont('arial', 50)
    tn_surface = timber_num_font.render(r'Timber: ' + str(timber_profit), False, (130, 182, 217))
    screen.blit(tn_surface, (50, 620))

    # the number of year and its font
    year_num = 0
    year_num_font = pygame.font.SysFont('arial', 50)
    year_num_font_surface = year_num_font.render(r'Year: ' + str(year_num), False, (130, 182, 217))
    screen.blit(year_num_font_surface, (400, 620))

    # create timber
    got_timber, got_timbers = False, False
    timber = RigidBody(os.getcwd() + r'\assets\timber.png')
    timber.set_acceleration(0.001)
    timbers = []  # store all timbers

    # get stump frames
    stump_frames = [
        pygame.image.load(os.getcwd() + r'\assets\PixelTrees\gif\stump\tile000.png').convert_alpha(),
        pygame.image.load(os.getcwd() + r'\assets\PixelTrees\gif\stump\tile001.png').convert_alpha(),
        pygame.image.load(os.getcwd() + r'\assets\PixelTrees\gif\stump\tile002.png').convert_alpha(),
        pygame.image.load(os.getcwd() + r'\assets\PixelTrees\gif\stump\tile003.png').convert_alpha(),
        pygame.image.load(os.getcwd() + r'\assets\PixelTrees\gif\stump\tile004.png').convert_alpha(),
        pygame.image.load(os.getcwd() + r'\assets\PixelTrees\gif\stump\tile005.png').convert_alpha()
    ]

    # get tree frames
    tree_frames = [
        pygame.image.load(os.getcwd() + r'\assets\trees-blackland\tree4\tree4_00.png').convert_alpha(),
        pygame.image.load(os.getcwd() + r'\assets\trees-blackland\tree4\tree4_01.png').convert_alpha(),
        pygame.image.load(os.getcwd() + r'\assets\trees-blackland\tree4\tree4_02.png').convert_alpha(),
        pygame.image.load(os.getcwd() + r'\assets\trees-blackland\tree4\tree4_03.png').convert_alpha()
    ]

    # create trees
    trees = []
    for _ in range(10):
        trees.append([])
        for _ in range(10):
            trees[-1].append(Tree([
                os.getcwd() + r'\assets\trees-blackland\tree4\tree4_00.png',
                os.getcwd() + r'\assets\trees-blackland\tree4\tree4_01.png',
                os.getcwd() + r'\assets\trees-blackland\tree4\tree4_02.png',
                os.getcwd() + r'\assets\trees-blackland\tree4\tree4_03.png'
            ], random.randint(2, 10)))
    for i in range(10):
        for j in range(10):
            tree = trees[i][j]
            tree.age = float(random.randint(-1, 7))  # random generation
            if tree.age == -1.0:
                tree.is_chopped = True
                tree.set_frames(stump_frames)
            tree.resize(45, 45)
            tree.set_pos((i*60+7, j*60+4))
            screen.blit(tree.image, tree.rect)

    # create font of hint for tree age
    age_font = pygame.font.SysFont('arial', 10)
    for row in trees:
        for tree in row:
            af_surface = age_font.render(r'age: ' + str(tree.age), False, (130, 182, 180))
            screen.blit(af_surface, tree.rect.move(0, 40))

    # create select cursor
    select_cursor = AnimeObject([
        os.getcwd() + r'\assets\PixelTrees\gif\selectcursor\tile000.png',
        os.getcwd() + r'\assets\PixelTrees\gif\selectcursor\tile001.png',
        os.getcwd() + r'\assets\PixelTrees\gif\selectcursor\tile002.png',
        os.getcwd() + r'\assets\PixelTrees\gif\selectcursor\tile003.png',
        os.getcwd() + r'\assets\PixelTrees\gif\selectcursor\tile004.png',
        os.getcwd() + r'\assets\PixelTrees\gif\selectcursor\tile005.png'
    ])
    select_cursor.resize(65, 65)
    select_cursor.set_pos((-3, -3))
    select_cursor.draw(screen)

    pygame.display.flip()

    # game loop
    while True:
        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if select_cursor.rect.top >= 50:
                        select_cursor.move(0, -60)
                elif event.key == pygame.K_DOWN:
                    if select_cursor.rect.bottom <= 600:
                        select_cursor.move(0, 60)
                elif event.key == pygame.K_LEFT:
                    if select_cursor.rect.left >= 50:
                        select_cursor.move(-60, 0)
                elif event.key == pygame.K_RIGHT:
                    if select_cursor.rect.right <= 600:
                        select_cursor.move(60, 0)
                elif event.key == pygame.K_c:  # chop down tree
                    row_num, col_num = round(select_cursor.rect.x / 60), round(select_cursor.rect.y / 60)
                    tree = trees[row_num][col_num]
                    if not tree.is_chopped:
                        tree.set_frames(stump_frames)
                        tree.is_chopped = True
                        timber_value = tree.get_timber_value(tree.age)
                        tree.age = -1.0  # just represent there should not have age
                        got_timber = True
                        timber.set_speed(0.0)
                        timber.set_acceleration(0.001)
                        timber.set_pos((select_cursor.rect.x + 15, select_cursor.rect.y + 15))
                        timber.set_start_end_pos((select_cursor.rect.x + 15, select_cursor.rect.y + 15), (100, 630))
                elif event.key == pygame.K_0:  # chop down all trees with age 0
                    for row in trees:
                        for tree in row:
                            if tree.age == 0.0:
                                tree.set_frames(stump_frames)
                                tree.is_chopped = True
                                timber_value = tree.get_timber_value(0.0)
                                tree.age = -1.0
                                got_timbers = True
                                timbers.append(RigidBody(os.getcwd() + r'\assets\timber.png'))
                                timber = timbers[-1]
                                timber.set_acceleration(0.001)
                                timber.set_pos((tree.rect.x + 15, tree.rect.y + 15))
                                timber.set_start_end_pos((tree.rect.x + 15, tree.rect.y + 15),
                                                         (100, 630))
                elif event.key == pygame.K_1:  # chop down all trees with age 1
                    for row in trees:
                        for tree in row:
                            if tree.age == 1.0:
                                tree.set_frames(stump_frames)
                                tree.is_chopped = True
                                timber_value = tree.get_timber_value(1.0)
                                tree.age = -1.0
                                got_timbers = True
                                timbers.append(RigidBody(os.getcwd() + r'\assets\timber.png'))
                                timber = timbers[-1]
                                timber.set_acceleration(0.001)
                                timber.set_pos((tree.rect.x + 15, tree.rect.y + 15))
                                timber.set_start_end_pos((tree.rect.x + 15, tree.rect.y + 15),
                                                         (100, 630))
                elif event.key == pygame.K_2:  # chop down all trees with age 2
                    for row in trees:
                        for tree in row:
                            if tree.age == 2.0:
                                tree.set_frames(stump_frames)
                                tree.is_chopped = True
                                timber_value = tree.get_timber_value(2.0)
                                tree.age = -1.0
                                got_timbers = True
                                timbers.append(RigidBody(os.getcwd() + r'\assets\timber.png'))
                                timber = timbers[-1]
                                timber.set_acceleration(0.001)
                                timber.set_pos((tree.rect.x + 15, tree.rect.y + 15))
                                timber.set_start_end_pos((tree.rect.x + 15, tree.rect.y + 15),
                                                         (100, 630))
                elif event.key == pygame.K_3:  # chop down all trees with age 3
                    for row in trees:
                        for tree in row:
                            if tree.age == 3.0:
                                tree.set_frames(stump_frames)
                                tree.is_chopped = True
                                timber_value = tree.get_timber_value(3.0)
                                tree.age = -1.0
                                got_timbers = True
                                timbers.append(RigidBody(os.getcwd() + r'\assets\timber.png'))
                                timber = timbers[-1]
                                timber.set_acceleration(0.001)
                                timber.set_pos((tree.rect.x + 15, tree.rect.y + 15))
                                timber.set_start_end_pos((tree.rect.x + 15, tree.rect.y + 15),
                                                         (100, 630))
                elif event.key == pygame.K_4:  # chop down all trees with age 4
                    for row in trees:
                        for tree in row:
                            if tree.age == 4.0:
                                tree.set_frames(stump_frames)
                                tree.is_chopped = True
                                timber_value = tree.get_timber_value(4.0)
                                tree.age = -1.0
                                got_timbers = True
                                timbers.append(RigidBody(os.getcwd() + r'\assets\timber.png'))
                                timber = timbers[-1]
                                timber.set_acceleration(0.001)
                                timber.set_pos((tree.rect.x + 15, tree.rect.y + 15))
                                timber.set_start_end_pos((tree.rect.x + 15, tree.rect.y + 15),
                                                         (100, 630))
                elif event.key == pygame.K_5:  # chop down all trees with age 5
                    for row in trees:
                        for tree in row:
                            if tree.age == 5.0:
                                tree.set_frames(stump_frames)
                                tree.is_chopped = True
                                timber_value = tree.get_timber_value(5.0)
                                tree.age = -1.0
                                got_timbers = True
                                timbers.append(RigidBody(os.getcwd() + r'\assets\timber.png'))
                                timber = timbers[-1]
                                timber.set_acceleration(0.001)
                                timber.set_pos((tree.rect.x + 15, tree.rect.y + 15))
                                timber.set_start_end_pos((tree.rect.x + 15, tree.rect.y + 15),
                                                         (100, 630))
                elif event.key == pygame.K_6:  # chop down all trees with age 6
                    for row in trees:
                        for tree in row:
                            if tree.age == 6.0:
                                tree.set_frames(stump_frames)
                                tree.is_chopped = True
                                timber_value = tree.get_timber_value(6.0)
                                tree.age = -1.0
                                got_timbers = True
                                timbers.append(RigidBody(os.getcwd() + r'\assets\timber.png'))
                                timber = timbers[-1]
                                timber.set_acceleration(0.001)
                                timber.set_pos((tree.rect.x + 15, tree.rect.y + 15))
                                timber.set_start_end_pos((tree.rect.x + 15, tree.rect.y + 15),
                                                         (100, 630))
                elif event.key == pygame.K_7:  # chop down all trees with age 7
                    for row in trees:
                        for tree in row:
                            if tree.age == 7.0:
                                tree.set_frames(stump_frames)
                                tree.is_chopped = True
                                timber_value = tree.get_timber_value(7.0)
                                tree.age = -1.0
                                got_timbers = True
                                timbers.append(RigidBody(os.getcwd() + r'\assets\timber.png'))
                                timber = timbers[-1]
                                timber.set_acceleration(0.001)
                                timber.set_pos((tree.rect.x + 15, tree.rect.y + 15))
                                timber.set_start_end_pos((tree.rect.x + 15, tree.rect.y + 15),
                                                         (100, 630))
                elif event.key == pygame.K_n:  # to next year
                    # all trees that are not chopped get one year older, except for those reached the maximal age,
                    # they keep no change. Trees with maximal age will seed around itself, these seeds will influence
                    # the soil without tree and new trees will grow up on those. However, soil in which there are trees
                    # alive will not be influenced by seeds.
                    year_num += 1
                    soil_with_seeds_pos = set()  # a set consists of all positions of soil with seeds
                    for i, row in enumerate(trees):
                        for j, tree in enumerate(row):
                            if not tree.is_chopped:
                                if tree.age < tree.maximal_age:
                                    tree.age += 1
                                else:
                                    # get all adjacent positions of this tree, because this tree will seed around itself
                                    soil_with_seeds_pos.update(
                                        {
                                            (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                                            (i, j - 1), (i, j + 1),
                                            (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)
                                        }
                                    )
                    soil_with_seeds_pos = \
                        filter(lambda soil_pos: 0 <= soil_pos[0] < 10 and 0 <= soil_pos[1] < 10, soil_with_seeds_pos)
                    for i, j in soil_with_seeds_pos:  # new trees will occur in these soil(premise: without tree)
                        tree = trees[i][j]
                        if tree.is_chopped:
                            tree.is_chopped = False
                            tree.age = 0.0
                            tree.set_frames(tree_frames)


        dt = clock.tick(60)
        t = pygame.time.get_ticks()

        # render background
        screen.blit(background, background.get_rect())

        # render fonts under forest map
        screen.fill((0, 0, 0), (0, 600, 600, 100))
        tn_surface = timber_num_font.render(r'Profit: ' + str(timber_profit), False, (130, 182, 217))
        screen.blit(tn_surface, (50, 620))

        # render fonts of the number of the year
        year_num_font_surface = year_num_font.render(r'Year: ' + str(year_num), False, (130, 182, 217))
        screen.blit(year_num_font_surface, (400, 620))

        # render trees
        for row in trees:
            for tree in row:
                tree.update(dt, t)
                tree.draw(screen)
                # age hint for each tree
                af_surface = age_font.render(r'age: ' + str(tree.age), False, (130, 182, 180))
                screen.blit(af_surface, tree.rect.move(0, 40))


        # timber slide
        if got_timber:
            timber.smooth_slide(dt)
            screen.blit(timber.image, timber.rect)
            if timber.rect.center[1] > 620:
                got_timber = False
                timber_profit += timber_value
        elif got_timbers:
            for timber in timbers:
                timber.smooth_slide(dt)
                screen.blit(timber.image, timber.rect)
                if timber.rect.center[1] > 610:
                    timbers.remove(timber)
                    timber_profit += timber_value
            if len(timbers) == 0:
                got_timbers = False

        # render select cursor
        select_cursor.update(dt, t)
        select_cursor.draw(screen)

        pygame.display.flip()
