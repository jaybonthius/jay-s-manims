from manim import *
import math
import numpy as np
import random
import os.path

config.background_color = WHITE
text_color = BLACK

# a colorblind-friendly color palette
# src: https://www.nature.com/articles/nmeth.1618
ORANGE = "#E69F00"
SKY_BLUE = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILION = "#D55E00"
MAUVE = "#CC79A7"


class SampleProportion( Scene ):
    def construct(self):

        show_population = True
        show_sampling_distribution = True
        show_sampling = True
        show_fancy_first_sample = True
        show_normal_curve = True
        play_sounds = True
        reuse_population = True
        reuse_samples = True

        wait_time = 1
        clack_volume = 0.1

        # * Making the population
        p = 0.25
        n = 3000
        x_dim = 12
        y_dim = 5

        # the larger the packing factor, the closer the dots (and the larger the radius)
        packing_factor = 1.5
        pop_radius = np.sqrt( packing_factor * x_dim * y_dim / n ) / 4
        min_pop_padding = 0.1 * pop_radius
        population = VGroup()

        pop_coords_file_name = os.path.join( "data",
                                             "n_" + str( n ) + "_p_" + str( round( 100 * p ) ).rstrip( '0' ).rstrip(
                                                 '.' ) + "_pop_values.csv" )
        pop_coords_values = np.empty( [n, 3] )
        if not reuse_population or not os.path.isfile( pop_coords_file_name ):
            i = 0
            while i < n:
                x_coord = x_dim * (random.uniform( -0.5, 0.5 ))
                y_coord = y_dim * (random.uniform( -0.5, 0.5 ))

                if is_there_space( [x_coord, y_coord], pop_coords_values, 2 * pop_radius + 2 * min_pop_padding ):
                    pop_coords_values[i][0] = x_coord
                    pop_coords_values[i][1] = y_coord
                    # makes proportion p of the points red
                    if i < n * p:
                        population.add( Dot( radius=pop_radius, fill_opacity=1, color=VERMILION ).move_to(
                            (x_coord * RIGHT + y_coord * UP) ) )
                        pop_coords_values[i][2] = 1
                    # makes proportion 1-p of the points blue
                    else:
                        population.add( Dot( radius=pop_radius, fill_opacity=1, color=LIGHT_GRAY ).move_to(
                            (x_coord * RIGHT + y_coord * UP) ) )
                        pop_coords_values[i][2] = 0
                    i = i + 1
                    if i % 10 == 0:
                        print( i, end='\r' )
            print()
            np.savetxt( pop_coords_file_name, pop_coords_values, delimiter=',' )
        else:
            print(
                "Nice! Population coordinates are already made! If you want to generate new one, make `reuse = False`." )
            pop_coords_values = np.loadtxt( pop_coords_file_name, delimiter=',' )
            i = 0
            while i < n * p:
                population.add( Dot( radius=pop_radius, fill_opacity=1, color=VERMILION ).move_to(
                    (pop_coords_values[i][0] * RIGHT + pop_coords_values[i][1] * UP) ) )
                i += 1
            while i < n:
                population.add( Dot( radius=pop_radius, fill_opacity=1, color=LIGHT_GRAY ).move_to(
                    (pop_coords_values[i][0] * RIGHT + pop_coords_values[i][1] * UP) ) )
                i += 1

        pop_rect = RoundedRectangle( height=y_dim + 0.25, width=x_dim + 0.25, corner_radius=0.25, color=GRAY,
                                     fill_opacity=0 )
        pop_group = VGroup( pop_rect, population )
        pop_string = "p = " + str( p )
        pop_text = MathTex( pop_string, color=text_color ).next_to( pop_rect, DOWN )

        if show_population:
            self.play(
                Create( population ),
                Create( pop_rect )
            )
            self.play(
                Write( pop_text )
            )
            self.play( Wait( wait_time ) )
            # self.wait()
            self.play( pop_group.animate.scale( 0.5 ).move_to( UP * 2 ), FadeOut( pop_text ) )
            self.play( Wait( wait_time ) )

        # ####################
        # HELPER SCENES
        # ####################
        def sampling():
            sample_indices = np.empty( k, dtype='int' )
            point_estimate = 0
            for idx in range( k ):
                occupied = True
                candidate_index = 0
                while occupied:
                    candidate_index = random.randint( 0, n - 1 )
                    if candidate_index not in sample_indices:
                        sample_indices[idx] = candidate_index
                        occupied = False
                point_estimate += pop_coords_values[candidate_index][2]
            point_estimate = point_estimate / k
            return sample_indices, point_estimate

        def sampling_anim(sample_list, coords, idx, p_hat, show_sample, rt):
            x = coords[0][idx]
            y = coords[1][idx]
            cols = 20
            col_tracker = 0
            dist_between_cols = 1 / 5
            dist_between_rows = 1 / 5
            grab_the_dots = []
            sample_array = []

            for j in range( len( sample_list ) ):
                temp_sample = population[sample_list[j]].copy()
                sample_array.append( temp_sample )
                grab_the_dots.append(
                    sample_array[j].animate( run_time=rt ).set_width( 0.1 ).move_to(
                        LEFT * (cols / 2) * dist_between_cols + RIGHT * (
                                j % cols) * dist_between_cols + dist_between_rows * DOWN * math.floor(
                            col_tracker / cols ) ) )
                col_tracker += 1

            vg = VGroup()
            for j in range( len( sample_list ) ):
                vg.add( sample_array[j] )

            dt = Dot( radius=radius, color=GREEN ).move_to( x * RIGHT + y * UP )
            if show_sample:
                self.play( *grab_the_dots )
                if idx == 0 and show_fancy_first_sample:
                    a_sample_text = Tex( "A \\textbf{random sample}", "\\\\ of size " + str( k ) ).next_to( vg,
                                                                                                            2 * DOWN ).set_color(
                        MAUVE )
                    self.wait( 0.5 )
                    self.play( Circumscribe( vg, color=MAUVE ), Write( a_sample_text[0] ) )
                    self.play( Write( a_sample_text[1] ) )
                    self.play( Wait( wait_time ) )
                    self.play( FadeOut( a_sample_text ) )
                    success_dots = VGroup()
                    successes = 0
                    for j in range( len( sample_list ) ):
                        if pop_coords_values[sample_list[j]][2] == 1:
                            success_dots.add( sample_array[j] )
                            successes += 1
                    success_text = MathTex( str( successes ), r"\text{ successes}", color=text_color )
                    success_text[0].set_color( VERMILION )
                    total_text = MathTex( r"\text{out of }", str( len( sample_list ) ), r"\text{ total}",
                                          color=text_color )
                    # total_text[1].set_color( GRAY )
                    text = VGroup( success_text, total_text ).next_to( vg, 2 * RIGHT )
                    text.arrange( DOWN, center=False, aligned_edge=LEFT )
                    self.play( Indicate( success_dots, color=VERMILION ), Write( success_text ) )
                    self.play( Wait( wait_time ) )
                    self.play( Indicate( vg, color=GRAY ), Write( total_text ) )
                    self.play( Wait( wait_time ) )

                    p_hat_text = MathTex( r"\hat{p} = ", r"{" + str( successes ), r"\over",
                                          str( len( sample_list ) ) + r"}", color=text_color ).next_to( vg,
                                                                                                        2 * RIGHT )
                    p_hat_text[1].set_color( VERMILION )
                    # p_hat_text[3].set_color( GRAY )
                    self.play( ReplacementTransform( success_text[0], p_hat_text[1] ),
                               ReplacementTransform( total_text[1], p_hat_text[3] ), Write( p_hat_text[2] ),
                               FadeOut( success_text[1] ), FadeOut( total_text[0] ), FadeOut( total_text[2] ) )
                    self.play( Wait( wait_time ) )

                    p_hat_decimal = MathTex( str( round( p_hat, 3 ) ), color=text_color ).move_to(
                        p_hat_text[2].get_center() )
                    self.play( Write( p_hat_text[0] ) )
                    self.play( Wait( wait_time ) )
                    self.play( Transform( p_hat_text[1:], p_hat_decimal ) )
                    self.play( Wait( wait_time ) )
                    self.play( Uncreate( vg ), p_hat_text.animate().move_to( vg.get_center() ) )
                    pe_text = Tex( r"A \textbf{point estimate}" ).next_to( p_hat_text, 2 * RIGHT ).set_color(
                        MAUVE )
                    self.play( Wait( wait_time ) )
                    self.play( Circumscribe( p_hat_text, color=MAUVE ), Write( pe_text ) )
                    self.play( Wait( wait_time ) )
                    self.play( FadeOut( pe_text ), Create( axis ), Write( axis_labels ) )
                    self.add_foreground_mobject( axis )
                    self.play( Wait( wait_time ) )
                else:
                    if i == 0:
                        self.play( Create( axis ), Write( axis_labels ) )
                        self.add_foreground_mobject( axis )
                    p_hat_text = MathTex( r"\hat{p} = ", str( round( p_hat, 3 ) ), color=text_color ).shift(
                        ((len( sample_list ) / cols - 1) / 2) * dist_between_cols * DOWN )
                    self.play( Uncreate( vg, run_time=rt ), Create( p_hat_text, run_time=rt ) )
                if play_sounds:
                    self.add_sound( "click.wav", time_offset=1, gain=clack_volume )
                self.play( ReplacementTransform( p_hat_text, dt ) )
            elif rt > 0.07:
                if play_sounds:
                    self.add_sound( "click.wav", time_offset=rt, gain=clack_volume )
                self.play( ReplacementTransform( vg, dt, run_time=rt ) )
            else:
                if play_sounds:
                    self.add_sound( "click.wav", time_offset=rt, gain=clack_volume )
                self.play( Create( dt, run_time=rt ) )

        # * Sampling
        if show_sampling_distribution:
            # k = sample size
            k = 180
            # m = number of point estimates
            m = 300
            meta_sample_list = np.empty( [m, k], dtype='int' )
            p_hats = np.empty( m )
            meta_sample_list_filename = os.path.join( "data",
                                                      "k_" + str( k ) + "_m_" + str( m ) + "_meta_sample_list.csv" )
            phats_filename = os.path.join( "data", "k_" + str( k ) + "_m_" + str( m ) + "_phats.csv" )
            if not reuse_samples or not reuse_population or not os.path.isfile( meta_sample_list_filename ):
                for i in range( m ):
                    temp = sampling()
                    meta_sample_list[i] = temp[0]
                    p_hats[i] = temp[1]
                np.savetxt( meta_sample_list_filename, meta_sample_list, delimiter=',', fmt='%i' )
                np.savetxt( phats_filename, p_hats, delimiter=',' )
            else:
                print(
                    "Nice! Sampling coords are already made! If you want to generate new one, make `reuse_samples = False`." )
                meta_sample_list = np.loadtxt( meta_sample_list_filename, delimiter=',', dtype='int' )
                p_hats = np.loadtxt( phats_filename, delimiter=',' )

            #
            spaced_out = True
            a, b = 0.1, 0.4
            radius = min( (0.3 / m) ** (2. / 5), 0.1 )
            padding = 0.1 * radius
            axis_height = -3.5
            axis = Line( [-8, axis_height, 0], [8, axis_height, 0], color=text_color )
            axis_labels = VGroup()

            if spaced_out:
                for i in range( 11 ):
                    axis_labels.add(
                        DecimalNumber( i / 10, show_ellipsis=False, num_decimal_places=2, include_sign=False,
                                       color=text_color ).scale(
                            0.6 ).next_to( axis, 0.75 * DOWN ).shift(
                            (2 * radius + 2 * padding) * (i / 10 - p) * k * RIGHT ) )
            else:
                for i in range( 11 ):
                    axis_labels.add(
                        DecimalNumber( i / 10, show_ellipsis=False, num_decimal_places=2, include_sign=False,
                                       color=text_color ).scale(
                            0.6 ).next_to( axis, 0.75 * DOWN ).shift(
                            (-6 + 12 * (i / 10 - a) / (b - a)) * RIGHT ) )

            reuse_population = False
            point_estimate_coords = coords_generator( m, k, axis_height, radius, padding, p_hats, p, spaced_out, a, b )
            dot_array = []
            # pause = how many samples to show
            pause = 7
            # limit = how may total point estimates to show
            # limit = m
            limit = 30
            if show_sampling:
                speed = 1
                for i in range( min( pause, limit ) ):
                    speed = 0.2 + 0.8 / (1 + i ** 2)
                    sampling_anim( meta_sample_list[i], point_estimate_coords, i, p_hats[i], True,
                                   speed )
                last_speed = speed
                for i in range( pause, max( pause, limit ) ):
                    speed = 0.025 + (3 * last_speed) / (1 + (i - pause) ** 2)
                    sampling_anim( meta_sample_list[i], point_estimate_coords, i, p_hats[i], False,
                                   speed )
            else:
                self.add( axis, axis_labels )
                self.add_foreground_mobject( axis )
                for i in range( m ):
                    dot_array.append(
                        Dot( radius=radius, color=GREEN ).move_to(
                            point_estimate_coords[0][i] * RIGHT + point_estimate_coords[1][i] * UP ) )
                    self.add( dot_array[i] )
                self.wait( 1 )

            if show_normal_curve:
                se = np.sqrt( p * (1 - p) / k )

                ax = Axes(
                    x_range=[0, 0.5, 0.1],
                    y_range=[0, 25, 1],
                    x_length=12.5,
                    tips=False,
                    axis_config={"include_numbers": False}
                ).set_opacity( 1 ).shift( 0.5 * DOWN )

                dist = ax.get_graph(
                    lambda x: (1 / (se * np.sqrt( 2 * np.pi ))) * np.exp( (-1 / 2) * (((x - p) / se) ** 2) ),
                    x_range=[0, 0.5], use_smoothing=True ).set_fill( GREEN )
                dist_area = ax.get_area( graph=dist, x_range=(0, 0.5) ).set_opacity( 0.75 )
                self.play( Create( dist ), FadeIn( dist_area ) )
                line = DashedLine( [0, axis_height, 0], [0, -0.25, 0] ).set_color( BLACK )
                center_text = MathTex( r"p = ", str( p ), color=text_color ).next_to( line, UP )
                self.play( Create( line ), Write( center_text ) )
                self.wait()

        else:
            return


# ######################
# HELPER FUNCTIONS
# ######################
def norm(a, b):
    """
    :param a: an ordered pair in list form
    :param b: an ordered pair in list form
    :return: the Euclidean distance between points a and b
    """
    return math.sqrt( (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 )


def is_there_space(p, points, dist):
    """
    :param p: an ordered pair in list form
    :param points: a list of ordered pairs in list form
    :param dist:
    :return:
    """
    for q in points:
        # if q is within dist units of this point
        if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < dist ** 2:
            # print( (P[0] - Q[0]) ** 2, " + ", (P[1] - Q[1]) ** 2, " < ", dist ** 2 )
            return False
    return True


def coords_generator(m, k, ground, radius, padding, x_coords, p, spaced_out, a, b):
    coords = np.empty( [2, m] )
    # this makes the dots EQUALLY SPACED OUT
    if spaced_out:
        for i in range( m ):
            coords[0][i] = (2 * radius + 2 * padding) * (x_coords[i] - p) * k
    else:
        for i in range( m ):
            coords[0][i] = -6 + 12 * (x_coords[i] - a) / (b - a)

    coords[1][0] = ground + radius
    nearby_index = []
    for i in range( 1, m ):
        print( 100 * i / m, "%", end='\r' )
        x = coords[0][i]
        for j in range( i ):
            if x - coords[0][j] <= 2 * radius + padding:
                nearby_index.append( j )
        y0 = ground + radius
        if not len( nearby_index ) == 0:
            done = False
            while not done:
                j = 0
                found_intersection = False
                while j < len( nearby_index ) and not found_intersection:
                    x1 = coords[0][nearby_index[j]]
                    y1 = coords[1][nearby_index[j]]
                    if norm( [x, y0], [x1, y1] ) < 2 * radius + padding:
                        found_intersection = True
                        del nearby_index[j]
                        y0 = math.sqrt( (2 * radius + 2 * padding) ** 2 - (x - x1) ** 2 ) + y1
                    else:
                        j += 1
                if not found_intersection:
                    done = True
        coords[1][i] = y0
        nearby_index.clear()
    return coords
