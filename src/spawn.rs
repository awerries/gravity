use colorgrad;
use rand::distributions::Uniform;
use rand::Rng;

use bevy::{prelude::*, sprite::MaterialMesh2dBundle, window::PrimaryWindow};

pub fn setup(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let window = windows.get_single().unwrap();
    commands.spawn(Camera2dBundle::default());

    let size_factor = 3;
    let f2 = size_factor * size_factor;

    let w = window.width() * crate::DSCALE;
    let num_children = 1;
    let parent_particles = crate::NUM_PARTICLES * (f2 - num_children) / f2;
    let child_particles = crate::NUM_PARTICLES / f2;
    spawn_galaxy(
        &mut commands,
        &mut meshes,
        &mut materials,
        parent_particles,
        w * crate::GALAXY_WIDTH_SCALE,
        Vec2::new(0.0, 0.0),
        Vec2::new(0.0, 0.0),
        1.0,
    );

    let y1 = w * crate::DSCALE / 3.0;
    let bmass = crate::PARTICLE_MASS_LOWER * parent_particles as f32 * crate::BLACK_HOLE_REL_MASS;
    let v = (crate::GRAVITY * bmass / (y1 * y1).sqrt()).sqrt();
    let v = v / 1.5; // skewed velocity to make it elliptical
    spawn_galaxy(
        &mut commands,
        &mut meshes,
        &mut materials,
        child_particles,
        w * crate::GALAXY_WIDTH_SCALE / size_factor as f32,
        Vec2::new(0.0, y1),
        Vec2::new(v, 0.0),
        1.0,
    );

    /*
    spawn_grid(
        &mut commands,
        &mut meshes,
        &mut materials,
        crate::NUM_PARTICLES,
        window.width() * crate::DSCALE,
        window.height() * crate::DSCALE,
    );
    */
}

#[allow(dead_code)]
pub fn spawn_grid(
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    mut materials: &mut ResMut<Assets<ColorMaterial>>,
    num_particles: u32,
    width: f32,
    height: f32,
) {
    let mut rng = rand::thread_rng();
    let xdist = Uniform::new(-width / 2.0, width / 2.0);
    let ydist = Uniform::new(-height / 2.0, height / 2.0);
    let vdist = Uniform::new(0., crate::VEL_VARIATION);
    let vthetadist = Uniform::new(0.0, 2.0 * std::f32::consts::PI);
    let mdist = Uniform::new(crate::PARTICLE_MASS_LOWER, crate::PARTICLE_MASS_UPPER);
    for ii in 0..num_particles {
        let mass = rng.sample(mdist);
        let v = rng.sample(vdist);
        let vtheta = rng.sample(vthetadist);

        spawn_particle(
            &mut commands,
            &mut meshes,
            &mut materials,
            (ii as f32) / (num_particles as f32),
            Vec2::new(rng.sample(xdist), rng.sample(ydist)),
            Vec2::new(v * vtheta.cos(), v * vtheta.sin()),
            mass,
            get_star_color(mass),
        );
    }
}

#[allow(dead_code)]
fn spawn_galaxy(
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    mut materials: &mut ResMut<Assets<ColorMaterial>>,
    num_particles: u32,
    diameter: f32,
    gpos: Vec2,
    gvel: Vec2,
    rotation: f32,
) {
    let bmass = crate::PARTICLE_MASS_LOWER * num_particles as f32 * crate::BLACK_HOLE_REL_MASS;
    let mut gmass = 0.; // previously: (num_particles as f32) * crate::AVG_PARTICLE_MASS;
    if crate::SPAWN_BLACKHOLES {
        gmass += bmass;
    }

    let mut rng = rand::thread_rng();
    let mdist = Uniform::new(crate::PARTICLE_MASS_LOWER, crate::PARTICLE_MASS_UPPER);
    for ii in 0..num_particles {
        let (r, theta, pos) = random_circle_pos(diameter);
        let vel = random_orbital_circle_vel(r, theta, gmass, rotation);
        let mass = rng.sample(mdist);

        spawn_particle(
            &mut commands,
            &mut meshes,
            &mut materials,
            (ii as f32) / (num_particles as f32),
            pos + gpos,
            vel + gvel,
            mass,
            get_star_color(mass),
        );
    }

    if crate::SPAWN_BLACKHOLES {
        spawn_particle(
            &mut commands,
            &mut meshes,
            &mut materials,
            0.0,
            gpos,
            gvel,
            bmass,
            Color::DARK_GRAY,
        );
    }
}

fn get_star_color(mass: f32) -> Color {
    let g = colorgrad::rd_yl_bu();
    let mass_ratio = (mass - crate::PARTICLE_MASS_LOWER)
        / (crate::PARTICLE_MASS_UPPER - crate::PARTICLE_MASS_LOWER);
    let mass_ratio_skewed = mass_ratio.sqrt().sqrt();
    let c = g.at(mass_ratio_skewed as f64).to_rgba8();
    Color::rgba_u8(c[0], c[1], c[2], c[3])
}

fn random_circle_pos(diameter: f32) -> (f32, f32, Vec2) {
    let mut rng = rand::thread_rng();
    let rdist = Uniform::new(0.00001, 1.0_f32);
    let r = (diameter / 2.0) * rng.sample(rdist).sqrt();

    let theta_dist = Uniform::new(0., 2.0 * std::f32::consts::PI);
    let theta = rng.sample(theta_dist) * 2. * std::f32::consts::PI;
    (r, theta, Vec2::new(r * theta.cos(), r * theta.sin()))
}

fn random_orbital_circle_vel(r: f32, theta: f32, mass: f32, rotation: f32) -> Vec2 {
    let mut rng = rand::thread_rng();
    let r = r + crate::MIN_R;
    let v = (crate::GRAVITY * mass / (r * r).sqrt()).sqrt();
    Vec2::new(
        rotation * v * theta.sin() + v * rng.gen_range(-crate::VEL_VARIATION..crate::VEL_VARIATION),
        rotation * -v * theta.cos()
            + v * rng.gen_range(-crate::VEL_VARIATION..crate::VEL_VARIATION),
    )
}

fn spawn_particle(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    index: f32,
    pos: Vec2,
    vel: Vec2,
    mass: f32,
    color: Color,
) {
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(1.0).into()).into(),
            material: materials.add(color.into()),
            transform: Transform::from_translation(Vec3::new(pos.x, pos.y, index) / crate::DSCALE),
            ..default()
        },
        crate::Pose {
            r: pos,
            v: vel,
            prev_accel: None,
        },
        crate::Mass(mass),
    ));
}
