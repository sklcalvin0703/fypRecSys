module.exports = {
  apps : [{
    name: 'fypRecSys',
    script: 'app.py',

    // Options reference: https://pm2.io/doc/en/runtime/reference/ecosystem-file/
    args: 'one two',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
	  HTTP_PORT: '80',
	  HTTPS_PORT: '443',
    },
    env_production: {
      NODE_ENV: 'production'
    }
  }],

  deploy : {
    production : {
      user : 'root',
      host : 'FYPBACKEND.MOOO.COM',
      ref  : 'origin/master',
      repo : 'git@github.com:sklcalvin0703/fypRecSys.git',
      path : '/home/root/production/fypBackend',
      'post-deploy' : 'pip install -r requirement.txt && pm2 start'
    }
  }
};
